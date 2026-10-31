use std::cmp;
use std::io::Write;
use std::time::Duration;

use utils::eventfd::EventFd;
use utils::metrics::MetricsWriter;
use utils::timerfd::TimerFd;
use vm_memory::{Address, ByteValued, GuestMemoryBackend, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, BalloonError, DeviceQueue, DeviceState, HostMemoryRange,
    QueueConfig, VirtioDevice, VirtioStateError,
};
use super::{defs, defs::uapi};
use crate::virtio::InterruptTransport;

// Inflate queue.
pub(crate) const IFQ_INDEX: usize = 0;
// Deflate queue.
pub(crate) const DFQ_INDEX: usize = 1;
// Stats queue.
pub(crate) const STQ_INDEX: usize = 2;
// Page-hinting queue.
pub(crate) const PHQ_INDEX: usize = 3;
// Free page reporting queue.
pub(crate) const FRQ_INDEX: usize = 4;

// Supported features.
pub(crate) const BASE_AVAIL_FEATURES: u64 = (1 << uapi::VIRTIO_F_VERSION_1 as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_STATS_VQ as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_FREE_PAGE_HINT as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_REPORTING as u64);

pub(crate) const VIRTIO_BALLOON_S_AVAIL: u16 = 6;
pub(crate) const MAX_STATS_TAGS: u32 = 256;
pub(crate) const MAX_STATS_DESC_LEN: u32 =
    MAX_STATS_TAGS * std::mem::size_of::<BalloonStat>() as u32;

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct VirtioBalloonConfig {
    /* Number of pages host wants Guest to give up. */
    num_pages: u32,
    /* Number of pages we've actually got in balloon. */
    actual: u32,
    /* Free page report command id, readonly by guest */
    free_page_report_cmd_id: u32,
    /* Stores PAGE_POISON if page poisoning is in use */
    poison_val: u32,
}

// Safe because it only has data and has no implicit padding.
unsafe impl ByteValued for VirtioBalloonConfig {}

#[derive(Clone, Copy, Default)]
#[repr(C, packed)]
pub(crate) struct BalloonStat {
    pub(crate) tag: vm_memory::Le16,
    pub(crate) val: vm_memory::Le64,
}

unsafe impl ByteValued for BalloonStat {}

pub struct Balloon {
    pub(crate) queues: Option<Vec<DeviceQueue>>,
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
    config: VirtioBalloonConfig,
    pub(crate) metrics: MetricsWriter,
    pub(crate) stats_polling_interval: Option<Duration>,
    pub(crate) stats_timer: TimerFd,
    pub(crate) stats_desc_index: Option<u16>,
}

impl Balloon {
    pub fn new(
        metrics: MetricsWriter,
        stats_polling_interval: Option<Duration>,
    ) -> super::Result<Balloon> {
        Ok(Balloon {
            queues: None,
            avail_features: BASE_AVAIL_FEATURES,
            acked_features: 0,
            activate_evt: EventFd::new(utils::eventfd::EFD_NONBLOCK)
                .map_err(BalloonError::EventFd)?,
            device_state: DeviceState::Inactive,
            config: VirtioBalloonConfig::default(),
            metrics,
            stats_polling_interval,
            stats_timer: TimerFd::new().map_err(BalloonError::TimerFd)?,
            stats_desc_index: None,
        })
    }

    pub fn id(&self) -> &str {
        defs::BALLOON_DEV_ID
    }

    pub(crate) fn stats_enabled(&self) -> bool {
        self.stats_polling_interval.is_some()
    }

    pub fn process_frq(&mut self) -> bool {
        debug!("balloon: process_frq()");
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        // Only a driver that accepted free page reporting may hand back pages through its queue.
        if self.acked_features & (1 << uapi::VIRTIO_BALLOON_F_REPORTING as u64) == 0 {
            warn!("balloon: free page report without the reporting feature negotiated, ignored");
            return false;
        }

        let queues = self
            .queues
            .as_mut()
            .expect("queues should exist when activated");
        let mut have_used = false;

        while let Some(head) = queues[FRQ_INDEX].queue.pop(mem) {
            let index = head.index;
            for desc in head.into_iter() {
                // The range comes from the guest. Discarding one that runs past the end of guest
                // memory would discard whatever the VMM has mapped after it, so every byte must
                // be guest RAM. A report may cross from one region into an adjacent one.
                let Ok(slices) = mem
                    .get_slices(desc.addr, desc.len as usize)
                    .collect::<Result<Vec<_>, _>>()
                else {
                    warn!(
                        "balloon: ignoring free page report outside guest memory: guest_addr={:?} len={}",
                        desc.addr, desc.len
                    );
                    continue;
                };
                if !queues[FRQ_INDEX].queue.mark_host_write(HostMemoryRange {
                    start: desc.addr.raw_value(),
                    length: u64::from(desc.len),
                }) {
                    error!("balloon: free-page range was not covered by an admitted request");
                    continue;
                }
                for slice in &slices {
                    let guard = slice.ptr_guard_mut();
                    debug!(
                        "balloon: should release guest_addr={:?} host_addr={:p} len={}",
                        desc.addr,
                        guard.as_ptr(),
                        guard.len()
                    );
                    if let Err(e) = discard_guest_pages(guard.as_ptr(), guard.len()) {
                        error!("balloon: failed to discard reported free pages: {e:?}");
                    }
                }
            }

            have_used = true;
            if let Err(e) = queues[FRQ_INDEX].queue.add_used(mem, index, 0) {
                error!("failed to add used elements to the queue: {e:?}");
            }
        }

        have_used
    }
}

#[cfg(unix)]
fn discard_guest_pages(host_addr: *mut u8, len: usize) -> std::io::Result<()> {
    let ret = unsafe { libc::madvise(host_addr as *mut libc::c_void, len, libc::MADV_DONTNEED) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn discard_guest_pages(host_addr: *mut u8, len: usize) -> std::io::Result<()> {
    // The guest supplied these ranges through virtio-balloon free-page reporting, so the host can
    // discard its resident backing pages without changing the guest-visible address space.
    unsafe { crate::windows::memory_mapping::discard_virtual_memory_range(host_addr.cast(), len) }
}

impl VirtioDevice for Balloon {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features
    }

    fn device_type(&self) -> u32 {
        uapi::VIRTIO_ID_BALLOON
    }

    fn device_name(&self) -> &str {
        "balloon"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &defs::QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config_slice = self.config.as_slice();
        let config_len = config_slice.len() as u64;
        if offset >= config_len {
            error!("Failed to read config space");
            return;
        }
        if let Some(end) = offset.checked_add(data.len() as u64) {
            // This write can't fail, offset and end are checked against config_len.
            data.write_all(&config_slice[offset as usize..cmp::min(end, config_len) as usize])
                .unwrap();
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "balloon: guest driver attempted to write device config (offset={:x}, len={:x})",
            offset,
            data.len()
        );
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        if queues.len() != defs::NUM_QUEUES {
            error!(
                "Cannot perform activate. Expected {} queue(s), got {}",
                defs::NUM_QUEUES,
                queues.len()
            );
            return Err(ActivateError::BadActivate);
        }

        if self.activate_evt.write(1).is_err() {
            error!("Cannot write to activate_evt",);
            return Err(ActivateError::BadActivate);
        }

        self.queues = Some(queues);
        self.device_state = DeviceState::Activated(mem, interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        self.stats_desc_index = None;
        let _ = self.stats_timer.disarm();
        self.queues = None;
        self.device_state = DeviceState::Inactive;
        true
    }

    fn supports_quiesce(&self) -> bool {
        true
    }

    fn quiesce(&mut self) -> Result<Vec<DeviceQueue>, VirtioStateError> {
        if !self.device_state.is_activated() {
            return Err(VirtioStateError::InvalidLifecycle(
                "balloon must be activated before quiescence",
            ));
        }

        // The stats queue deliberately retains one consumed descriptor between samples. Return it
        // through the used ring so the captured queue satisfies consumed-or-terminal ownership.
        self.trigger_stats_update();
        self.stats_timer.disarm()?;
        let queues = self
            .queues
            .take()
            .ok_or(VirtioStateError::InvalidLifecycle(
                "balloon is activated without queues",
            ))?;
        self.device_state = DeviceState::Inactive;
        Ok(queues)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::sync::Arc;

    use utils::eventfd::EventFd;
    use vm_memory::{GuestAddress, GuestRegionMmap, MmapRegion};

    use super::*;
    use crate::legacy::DummyIrqChip;
    use crate::virtio::queue::tests::VirtQueue;
    use crate::virtio::Queue;

    const PAGE: usize = 4096;
    const VIRTQ_DESC_F_WRITE: u16 = 0x2;
    const REPORTING: u64 = 1 << uapi::VIRTIO_BALLOON_F_REPORTING as u64;

    /// Three host pages in one mapping: the guest gets the first two as its RAM, and the third
    /// stands for whatever the VMM has mapped right after guest memory.
    struct Layout {
        host: *mut u8,
        mem: GuestMemoryMmap,
    }

    impl Layout {
        fn new() -> Self {
            // SAFETY: a fresh private anonymous mapping, unmapped in `drop`.
            let host = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    3 * PAGE,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(host, libc::MAP_FAILED);
            let host = host.cast::<u8>();
            // SAFETY: the first two pages lie in the mapping above, which outlives `mem`.
            let region = unsafe {
                MmapRegion::build_raw(
                    host,
                    2 * PAGE,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                )
            }
            .unwrap();
            let mem =
                GuestMemoryMmap::from_regions(vec![
                    GuestRegionMmap::new(region, GuestAddress(0)).unwrap()
                ])
                .unwrap();
            // SAFETY: both pages lie in the mapping.
            unsafe {
                std::ptr::write_bytes(host.add(PAGE), 0xa5, PAGE);
                std::ptr::write_bytes(host.add(2 * PAGE), 0x5a, PAGE);
            }
            Self { host, mem }
        }

        fn page(&self, index: usize) -> &[u8] {
            // SAFETY: index < 3 lies in the mapping.
            unsafe { std::slice::from_raw_parts(self.host.add(index * PAGE), PAGE) }
        }
    }

    impl Drop for Layout {
        fn drop(&mut self) {
            // SAFETY: the mapping made in `new`.
            unsafe { libc::munmap(self.host.cast(), 3 * PAGE) };
        }
    }

    /// Activates a balloon whose driver acked `features` and has put one free page report of
    /// `[addr, addr + len)` on the reporting queue, as the guest's `virtballoon_free_page_report`
    /// does.
    fn balloon_with_report(layout: &Layout, features: u64, addr: u64, len: u32) -> Balloon {
        let ring = VirtQueue::new(GuestAddress(0), &layout.mem, 16);
        ring.dtable[0].set(addr, len, VIRTQ_DESC_F_WRITE, 0);
        ring.avail.ring[0].set(0);
        ring.avail.idx.set(1);

        let mut balloon = Balloon::new(MetricsWriter::default(), None).unwrap();
        balloon.set_acked_features(features);
        let queues = (0..defs::NUM_QUEUES)
            .map(|index| {
                let queue = if index == FRQ_INDEX {
                    ring.create_queue()
                } else {
                    Queue::new(defs::QUEUE_SIZE)
                };
                DeviceQueue::new(queue, Arc::new(EventFd::new(0).unwrap()))
            })
            .collect();
        let interrupt =
            InterruptTransport::new(DummyIrqChip::new().into(), "balloon-test".into()).unwrap();
        balloon
            .activate(layout.mem.clone(), interrupt, queues)
            .unwrap();
        balloon
    }

    #[test]
    fn free_page_report_past_guest_memory_leaves_host_memory_alone() {
        let layout = Layout::new();
        // Starts in the last guest page and runs one page past the end of guest RAM.
        let mut balloon = balloon_with_report(&layout, REPORTING, PAGE as u64, 2 * PAGE as u32);

        assert!(
            balloon.process_frq(),
            "the report is still returned to the guest"
        );
        assert!(
            layout.page(2).iter().all(|&byte| byte == 0x5a),
            "the page after guest memory was discarded"
        );
        assert!(
            layout.page(1).iter().all(|&byte| byte == 0xa5),
            "a report that is not wholly guest RAM is ignored as a whole"
        );
    }

    #[test]
    fn free_page_report_inside_guest_memory_is_discarded() {
        let layout = Layout::new();
        let mut balloon = balloon_with_report(&layout, REPORTING, PAGE as u64, PAGE as u32);

        assert!(balloon.process_frq());
        assert!(
            layout.page(1).iter().all(|&byte| byte == 0),
            "the reported page reads as zeros once the host dropped it"
        );
        assert!(layout.page(2).iter().all(|&byte| byte == 0x5a));
    }

    #[test]
    fn free_page_report_without_the_reporting_feature_is_ignored() {
        let layout = Layout::new();
        let mut balloon = balloon_with_report(&layout, 0, PAGE as u64, PAGE as u32);

        assert!(!balloon.process_frq());
        assert!(layout.page(1).iter().all(|&byte| byte == 0xa5));
    }
}
