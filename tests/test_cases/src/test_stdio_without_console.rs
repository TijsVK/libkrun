//! The kernel hands init its stdio on /dev/console, but only if a console is
//! registered by then. The console behind hvc0 is a virtio-console port that
//! the guest driver adds asynchronously, so on a loaded host init can start
//! with fds 0-2 closed. Pointing the kernel console at a device that never
//! exists forces that case on every run: the workload must still get the
//! krun-stdout port as its stdout.

use macros::{guest, host};

pub struct TestStdioWithoutConsole;

#[host]
mod host {
    use super::*;

    use crate::common::setup_fs_and_enter;
    use crate::{krun_call, krun_call_u32};
    use crate::{Test, TestSetup};
    use krun_sys::*;

    impl Test for TestStdioWithoutConsole {
        fn start_vm(self: Box<Self>, test_setup: TestSetup) -> anyhow::Result<()> {
            unsafe {
                krun_call!(krun_set_log_level(KRUN_LOG_LEVEL_WARN))?;
                let ctx = krun_call_u32!(krun_create_ctx())?;
                krun_call!(krun_set_vm_config(ctx, 1, 256))?;
                krun_call!(krun_set_kernel_console(ctx, c"ttynone0".as_ptr()))?;
                setup_fs_and_enter(ctx, test_setup)?;
            }
            Ok(())
        }
    }
}

#[guest]
mod guest {
    use super::*;
    use crate::Test;

    impl Test for TestStdioWithoutConsole {
        fn in_guest(self: Box<Self>) {
            println!("OK");
        }
    }
}
