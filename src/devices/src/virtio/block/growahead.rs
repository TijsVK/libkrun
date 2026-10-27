// Copyright 2026 Microsandbox Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Grow-ahead allocation for sparse raw images on Linux hosts.
//!
//! The first write into a hole of a sparse image makes the host filesystem allocate blocks, and
//! the guest's next flush (`fdatasync`) then has to commit the host filesystem's journal as well
//! as write the data. A guest that syncs often (package managers, databases, `git`) pays that
//! commit for every newly touched block. Grow-ahead removes most of them: on the first guest
//! write into a chunk of the image that still has holes, the device writes zeros into those
//! holes first, so the chunk's later writes are overwrites of allocated blocks. Holes read as
//! zeros, so the guest-visible content never changes.
//!
//! The filled-but-unused space is capped per image: a chunk is only filled while the zeros the
//! guest has not yet written over stay at or under the cap, so a guest that scatters writes over
//! the whole disk cannot grow the image toward its full size. A guest discard (or write-zeroes)
//! in a chunk returns it to the unchecked state: the punched range is a hole again and the next
//! write may fill it again.

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;

use log::{debug, warn};

/// Bytes covered by one bit of a filled chunk's map: the common host filesystem block size.
const GRANULE: u64 = 4096;

/// Chunk size and cap of the grow-ahead policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrowAheadConfig {
    /// Size of the aligned chunk whose holes are filled on its first write.
    pub chunk_bytes: u64,
    /// Most filled bytes that the guest has not yet written over, per image.
    pub cap_bytes: u64,
}

impl Default for GrowAheadConfig {
    fn default() -> Self {
        Self {
            chunk_bytes: 64 * 1024 * 1024,
            cap_bytes: 256 * 1024 * 1024,
        }
    }
}

/// What is known about one chunk of the image.
enum Chunk {
    /// Not looked at since the device started or since a discard touched it.
    Unchecked,
    /// Holds no holes (or its holes were filled and the guest has since written all of them).
    Settled,
    /// Has `hole_bytes` of holes that did not fit under the cap when last looked at.
    Deferred { hole_bytes: u64 },
    /// Was filled; the map holds the granules the guest has not written yet.
    Filled(FillMap),
}

/// Granules of one chunk that hold fill zeros the guest has not written over.
struct FillMap {
    bits: Vec<u64>,
    count: u64,
}

impl FillMap {
    fn new(granules: u64) -> Self {
        Self {
            bits: vec![0; granules.div_ceil(64) as usize],
            count: 0,
        }
    }

    /// Marks granules `[first, last]` as filled; returns how many were newly marked.
    fn set(&mut self, first: u64, last: u64) -> u64 {
        let mut added = 0;
        for granule in first..=last {
            let (word, bit) = ((granule / 64) as usize, granule % 64);
            if self.bits[word] & (1 << bit) == 0 {
                self.bits[word] |= 1 << bit;
                added += 1;
            }
        }
        self.count += added;
        added
    }

    /// Clears granules `[first, last]`; returns how many were cleared.
    fn clear(&mut self, first: u64, last: u64) -> u64 {
        let mut removed = 0;
        for granule in first..=last {
            let (word, bit) = ((granule / 64) as usize, granule % 64);
            if self.bits[word] & (1 << bit) != 0 {
                self.bits[word] &= !(1 << bit);
                removed += 1;
            }
        }
        self.count -= removed;
        removed
    }
}

/// Grow-ahead state of one writable raw image.
pub(crate) struct GrowAhead {
    /// A handle on the image, used only to find holes (`SEEK_HOLE`/`SEEK_DATA`).
    file: File,
    config: GrowAheadConfig,
    chunks: Vec<Chunk>,
    /// Filled bytes (whole granules) the guest has not written over yet.
    unused_bytes: u64,
    /// Total bytes of zeros written by fills, for logs and tests.
    filled_bytes: u64,
    /// Set after a failed hole lookup or fill; the policy then stays off for this device.
    disabled: bool,
}

impl GrowAhead {
    pub(crate) fn new(file: File, config: GrowAheadConfig) -> io::Result<Self> {
        if config.chunk_bytes == 0 || !config.chunk_bytes.is_multiple_of(GRANULE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "grow-ahead chunk size must be a non-zero multiple of 4 KiB",
            ));
        }
        Ok(Self {
            file,
            config,
            chunks: Vec::new(),
            unused_bytes: 0,
            filled_bytes: 0,
            disabled: false,
        })
    }

    #[cfg(test)]
    pub(crate) fn unused_bytes(&self) -> u64 {
        self.unused_bytes
    }

    #[cfg(test)]
    pub(crate) fn filled_bytes(&self) -> u64 {
        self.filled_bytes
    }

    /// Fills the holes of every chunk that `[offset, offset + length)` touches for the first
    /// time, except the bytes the write itself covers. `fill(offset, length)` writes zeros.
    ///
    /// A failure is never the guest's: it is logged, the policy turns itself off for this
    /// device and the guest write goes ahead as it would without grow-ahead.
    pub(crate) fn before_write<F>(&mut self, offset: u64, length: u64, capacity: u64, mut fill: F)
    where
        F: FnMut(u64, u64) -> io::Result<()>,
    {
        if self.disabled || length == 0 {
            return;
        }
        let end = offset.saturating_add(length).min(capacity);
        let mut chunk_start = offset - offset % self.config.chunk_bytes;
        while chunk_start < end {
            let index = (chunk_start / self.config.chunk_bytes) as usize;
            let chunk_end = chunk_start
                .saturating_add(self.config.chunk_bytes)
                .min(capacity);
            if let Err(error) =
                self.fill_chunk(index, chunk_start, chunk_end, offset, end, &mut fill)
            {
                warn!("grow-ahead turned off for this disk after an error: {error}");
                self.disabled = true;
                return;
            }
            chunk_start = chunk_end;
        }
    }

    fn fill_chunk<F>(
        &mut self,
        index: usize,
        chunk_start: u64,
        chunk_end: u64,
        write_start: u64,
        write_end: u64,
        fill: &mut F,
    ) -> io::Result<()>
    where
        F: FnMut(u64, u64) -> io::Result<()>,
    {
        if self.chunks.len() <= index {
            self.chunks.resize_with(index + 1, || Chunk::Unchecked);
        }
        match self.chunks[index] {
            Chunk::Settled | Chunk::Filled(_) => return Ok(()),
            Chunk::Deferred { hole_bytes }
                if self.unused_bytes + hole_bytes > self.config.cap_bytes =>
            {
                return Ok(());
            }
            Chunk::Unchecked | Chunk::Deferred { .. } => {}
        }

        let ranges = subtract(
            find_holes(&self.file, chunk_start, chunk_end)?,
            write_start,
            write_end,
        );
        let mut map = FillMap::new((chunk_end - chunk_start).div_ceil(GRANULE));
        for &(start, end) in &ranges {
            map.set(
                (start - chunk_start) / GRANULE,
                (end - 1 - chunk_start) / GRANULE,
            );
        }
        // The granules the write covers are written by it at once, so they never count as unused.
        map.clear(
            (write_start.max(chunk_start) - chunk_start) / GRANULE,
            (write_end.min(chunk_end) - 1 - chunk_start) / GRANULE,
        );
        let fill_bytes = map.count * GRANULE;
        if ranges.is_empty() {
            self.chunks[index] = Chunk::Settled;
            return Ok(());
        }
        if self.unused_bytes + fill_bytes > self.config.cap_bytes {
            self.chunks[index] = Chunk::Deferred {
                hole_bytes: fill_bytes,
            };
            return Ok(());
        }

        // Settled before filling: a fill that fails part way must not be retried on every write.
        self.chunks[index] = Chunk::Settled;
        let mut written = 0;
        for &(start, end) in &ranges {
            fill(start, end - start)?;
            written += end - start;
        }
        self.filled_bytes += written;
        debug!(
            "grow-ahead filled {written} bytes of holes in chunk {index} ({} bytes unused now)",
            self.unused_bytes + fill_bytes
        );
        if map.count != 0 {
            self.unused_bytes += fill_bytes;
            self.chunks[index] = Chunk::Filled(map);
        }
        Ok(())
    }

    /// Records a completed guest write: its granules no longer count as filled but unused.
    pub(crate) fn after_write(&mut self, offset: u64, length: u64) {
        self.for_each_touched_chunk(offset, length, |this, index, first, last| {
            if let Chunk::Filled(map) = &mut this.chunks[index] {
                this.unused_bytes -= map.clear(first, last) * GRANULE;
                if map.count == 0 {
                    this.chunks[index] = Chunk::Settled;
                }
            }
        });
    }

    /// Records a discard or zeroing of `[offset, offset + length)`: the range may be holes again,
    /// so every chunk it touches goes back to unchecked and its fill accounting is dropped.
    pub(crate) fn forget(&mut self, offset: u64, length: u64) {
        self.for_each_touched_chunk(offset, length, |this, index, _, _| {
            if let Chunk::Filled(map) = &this.chunks[index] {
                this.unused_bytes -= map.count * GRANULE;
            }
            this.chunks[index] = Chunk::Unchecked;
        });
    }

    /// Calls `visit(self, chunk index, first granule, last granule)` for every known chunk that
    /// `[offset, offset + length)` touches, with the touched granules relative to the chunk.
    fn for_each_touched_chunk<V>(&mut self, offset: u64, length: u64, mut visit: V)
    where
        V: FnMut(&mut Self, usize, u64, u64),
    {
        if length == 0 {
            return;
        }
        let chunk_bytes = self.config.chunk_bytes;
        let end = offset.saturating_add(length);
        let mut chunk_start = offset - offset % chunk_bytes;
        while chunk_start < end {
            let index = (chunk_start / chunk_bytes) as usize;
            if index >= self.chunks.len() {
                return;
            }
            let chunk_end = chunk_start.saturating_add(chunk_bytes);
            let first = (offset.max(chunk_start) - chunk_start) / GRANULE;
            let last = (end.min(chunk_end) - 1 - chunk_start) / GRANULE;
            visit(self, index, first, last);
            chunk_start = chunk_end;
        }
    }
}

/// Returns the holes of `file` in `[start, end)` as sorted, disjoint `(start, end)` ranges.
fn find_holes(file: &File, start: u64, end: u64) -> io::Result<Vec<(u64, u64)>> {
    let mut holes = Vec::new();
    let mut position = start;
    while position < end {
        let Some(hole) = seek(file, position, libc::SEEK_HOLE)? else {
            break;
        };
        if hole >= end {
            break;
        }
        let data = seek(file, hole, libc::SEEK_DATA)?.unwrap_or(end).min(end);
        holes.push((hole, data));
        position = data;
    }
    Ok(holes)
}

/// `lseek(2)` with `SEEK_HOLE` or `SEEK_DATA`; `None` when no such offset exists at or after
/// `offset` (`ENXIO`).
fn seek(file: &File, offset: u64, whence: libc::c_int) -> io::Result<Option<u64>> {
    let offset = libc::off_t::try_from(offset)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    // SAFETY: lseek only reads the descriptor, which `file` keeps open for the call.
    let result = unsafe { libc::lseek(file.as_raw_fd(), offset, whence) };
    if result < 0 {
        let error = io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ENXIO) => Ok(None),
            _ => Err(error),
        };
    }
    Ok(Some(result as u64))
}

/// Removes `[cut_start, cut_end)` from sorted, disjoint ranges.
fn subtract(ranges: Vec<(u64, u64)>, cut_start: u64, cut_end: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::with_capacity(ranges.len() + 1);
    for (start, end) in ranges {
        if end <= cut_start || start >= cut_end {
            out.push((start, end));
            continue;
        }
        if start < cut_start {
            out.push((start, cut_start));
        }
        if end > cut_end {
            out.push((cut_end, end));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::os::unix::fs::FileExt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn sparse_image(name: &str, size: u64) -> (PathBuf, File) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "growahead-{name}-{}-{}.raw",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.set_len(size).unwrap();
        (path, file)
    }

    fn policy(file: &File, chunk: u64, cap: u64) -> GrowAhead {
        GrowAhead::new(
            file.try_clone().unwrap(),
            GrowAheadConfig {
                chunk_bytes: chunk,
                cap_bytes: cap,
            },
        )
        .unwrap()
    }

    /// Runs a guest write through the policy and the file, as the device does.
    fn guest_write(policy: &mut GrowAhead, file: &File, offset: u64, data: &[u8], capacity: u64) {
        let zeros = vec![0u8; MIB as usize];
        policy.before_write(offset, data.len() as u64, capacity, |at, length| {
            let mut done = 0;
            while done < length {
                let step = (length - done).min(MIB);
                file.write_all_at(&zeros[..step as usize], at + done)?;
                done += step;
            }
            Ok(())
        });
        file.write_all_at(data, offset).unwrap();
        policy.after_write(offset, data.len() as u64);
    }

    fn holes(file: &File, start: u64, end: u64) -> Vec<(u64, u64)> {
        find_holes(file, start, end).unwrap()
    }

    /// The tests need a temp directory on a filesystem that reports holes (tmpfs, ext4, xfs and
    /// btrfs do), or they would pass without filling anything.
    fn assert_reports_holes(file: &File, size: u64) {
        assert_eq!(
            holes(file, 0, size),
            vec![(0, size)],
            "the temp dir must report holes"
        );
    }

    #[test]
    fn the_first_write_into_a_chunk_fills_only_that_chunks_holes() {
        let (path, file) = sparse_image("first-write", 4 * MIB);
        assert_reports_holes(&file, 4 * MIB);
        let mut policy = policy(&file, MIB, 4 * MIB);

        guest_write(&mut policy, &file, MIB + 8192, &[0xab; 512], 4 * MIB);

        assert_eq!(
            holes(&file, MIB, 2 * MIB),
            vec![],
            "chunk 1 has no holes left"
        );
        assert_eq!(holes(&file, 0, MIB), vec![(0, MIB)], "chunk 0 untouched");
        assert_eq!(holes(&file, 2 * MIB, 4 * MIB), vec![(2 * MIB, 4 * MIB)]);
        assert_eq!(policy.filled_bytes(), MIB - 512);
        // The written granule is used; the other 255 are filled and unused.
        assert_eq!(policy.unused_bytes(), MIB - GRANULE);

        let mut buffer = vec![0xffu8; MIB as usize];
        file.read_exact_at(&mut buffer, MIB).unwrap();
        assert!(buffer[..8192].iter().all(|&b| b == 0));
        assert!(buffer[8192..8704].iter().all(|&b| b == 0xab));
        assert!(buffer[8704..].iter().all(|&b| b == 0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn data_already_in_a_chunk_is_never_overwritten() {
        let (path, file) = sparse_image("keeps-data", 2 * MIB);
        assert_reports_holes(&file, 2 * MIB);
        file.write_all_at(&[0x5a; 4096], 64 * 1024).unwrap();
        let mut policy = policy(&file, 2 * MIB, 4 * MIB);

        guest_write(&mut policy, &file, MIB, &[0x11; 4096], 2 * MIB);

        let mut kept = [0u8; 4096];
        file.read_exact_at(&mut kept, 64 * 1024).unwrap();
        assert_eq!(kept, [0x5a; 4096]);
        assert_eq!(holes(&file, 0, 2 * MIB), vec![]);
        assert_eq!(policy.filled_bytes(), 2 * MIB - 2 * 4096);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn the_cap_defers_fills_until_the_guest_has_used_filled_space() {
        let (path, file) = sparse_image("cap", 4 * MIB);
        assert_reports_holes(&file, 4 * MIB);
        // One chunk's fill fits under the cap, a second one does not.
        let mut policy = policy(&file, MIB, MIB + MIB / 2);

        guest_write(&mut policy, &file, 0, &[1; 4096], 4 * MIB);
        assert_eq!(policy.unused_bytes(), MIB - GRANULE);
        guest_write(&mut policy, &file, MIB, &[2; 4096], 4 * MIB);
        assert_eq!(
            holes(&file, MIB, 2 * MIB),
            vec![(MIB + 4096, 2 * MIB)],
            "chunk 1 deferred: only the guest's own block was written"
        );

        // The guest writes over most of chunk 0's zeros; chunk 1 now fits and fills on its next write.
        guest_write(
            &mut policy,
            &file,
            4096,
            &vec![3; (MIB - 2 * 4096) as usize],
            4 * MIB,
        );
        assert_eq!(policy.unused_bytes(), GRANULE);
        guest_write(&mut policy, &file, MIB + 8192, &[4; 4096], 4 * MIB);
        assert_eq!(holes(&file, MIB, 2 * MIB), vec![]);
        assert!(policy.unused_bytes() <= MIB + MIB / 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_discard_makes_the_chunk_fill_again_on_its_next_write() {
        let (path, file) = sparse_image("discard", 2 * MIB);
        assert_reports_holes(&file, 2 * MIB);
        let mut policy = policy(&file, MIB, 4 * MIB);
        guest_write(&mut policy, &file, 0, &[7; 4096], 2 * MIB);
        assert_eq!(holes(&file, 0, MIB), vec![]);

        // What the device's discard does to the image, then tells the policy.
        // SAFETY: fallocate only reads the open descriptor; the range is inside the file.
        let punched = unsafe {
            libc::fallocate(
                file.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                0,
                MIB as libc::off_t,
            )
        };
        assert_eq!(punched, 0);
        policy.forget(0, MIB);
        assert_eq!(policy.unused_bytes(), 0);
        assert_eq!(holes(&file, 0, MIB), vec![(0, MIB)]);

        guest_write(&mut policy, &file, 512 * 1024, &[8; 4096], 2 * MIB);
        assert_eq!(holes(&file, 0, MIB), vec![]);
        assert_eq!(policy.filled_bytes(), 2 * (MIB - 4096));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_write_across_two_chunks_fills_both_and_the_last_chunk_may_be_short() {
        let (path, file) = sparse_image("across", MIB + MIB / 2);
        assert_reports_holes(&file, MIB + MIB / 2);
        let capacity = MIB + MIB / 2;
        let mut policy = policy(&file, MIB, 4 * MIB);
        guest_write(&mut policy, &file, MIB - 4096, &[9; 8192], capacity);
        assert_eq!(holes(&file, 0, capacity), vec![]);
        assert_eq!(policy.filled_bytes(), capacity - 8192);
        assert_eq!(policy.unused_bytes(), capacity - 8192);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_failed_fill_turns_the_policy_off_and_leaves_the_write_alone() {
        let (path, file) = sparse_image("failure", 2 * MIB);
        assert_reports_holes(&file, 2 * MIB);
        let mut policy = policy(&file, MIB, 4 * MIB);
        let mut calls = 0;
        policy.before_write(0, 4096, 2 * MIB, |_, _| {
            calls += 1;
            Err(io::Error::other("disk full"))
        });
        policy.before_write(MIB, 4096, 2 * MIB, |_, _| {
            calls += 1;
            Ok(())
        });
        assert_eq!(calls, 1, "no fill is tried after the first failure");
        assert!(policy.disabled);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn subtract_cuts_the_write_out_of_the_holes() {
        assert_eq!(subtract(vec![(0, 100)], 10, 20), vec![(0, 10), (20, 100)]);
        assert_eq!(
            subtract(vec![(0, 10), (30, 40)], 10, 30),
            vec![(0, 10), (30, 40)]
        );
        assert_eq!(subtract(vec![(0, 100)], 0, 100), vec![]);
        assert_eq!(subtract(vec![(5, 15)], 0, 10), vec![(10, 15)]);
    }

    #[test]
    fn a_chunk_size_that_is_not_whole_granules_is_refused() {
        let (path, file) = sparse_image("config", MIB);
        let error = GrowAhead::new(
            file.try_clone().unwrap(),
            GrowAheadConfig {
                chunk_bytes: 1000,
                cap_bytes: MIB,
            },
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        std::fs::remove_file(path).unwrap();
    }
}
