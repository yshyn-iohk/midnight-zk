//! Per-phase memory instrumentation for the prover and keygen.
//!
//! [`log_phase`] emits a marker on the `midnight_bench` tracing target,
//! carrying resident-set size and the process high-water mark where the
//! platform can supply them. Paired `.start` / `.end` markers around each
//! phase show *which* phase is responsible for each step-up in peak memory,
//! which is the question that matters on a device with a hard ceiling — a
//! total says the run was too big, not what made it too big.
//!
//! Named `phase_metrics` rather than `bench` because upstream already has a
//! `plonk::bench` directory module behind the `bench-internal` feature, and the
//! two would collide.
//!
//! Ported from mailbox `0002`, which placed these in `prover.rs`. They live in
//! their own module here for the same reason `bases.rs` does: the macOS arm is
//! an `extern "C"` FFI declaration, and the crate denies `unsafe_code`
//! globally. Keeping the exception in one small, named file is better than an
//! `#[allow]` in the middle of the prover.
//!
//! Every target is covered, so this compiles everywhere including wasm — the
//! fallback simply reports no memory figures and the phase markers still emit.

/// Linux and Android: `/proc/self/status` gives both live RSS and the
/// high-water mark. A single read of a few KiB of kernel text, cheap enough to
/// call at every phase boundary without measurable wall-clock cost.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn sample_rss_hwm_kb() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss: Option<u64> = None;
    let mut hwm: Option<u64> = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss = v.split_whitespace().next().and_then(|n| n.parse().ok());
        } else if let Some(v) = line.strip_prefix("VmHWM:") {
            hwm = v.split_whitespace().next().and_then(|n| n.parse().ok());
        }
    }
    rss.zip(hwm)
}

/// macOS and iOS: there is no `/proc`, so read `getrusage(RUSAGE_SELF)`.
///
/// `ru_maxrss` is the high-water mark only, and in **bytes** on this platform
/// rather than Linux's KiB. There is no true "current RSS" available here, so
/// the same figure is returned for both and callers see them track together —
/// deliberately, so the return shape needs no branching at the call site.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(unsafe_code)]
fn sample_rss_hwm_kb() -> Option<(u64, u64)> {
    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i32,
    }
    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: i64,
        ru_ixrss: i64,
        ru_idrss: i64,
        ru_isrss: i64,
        ru_minflt: i64,
        ru_majflt: i64,
        ru_nswap: i64,
        ru_inblock: i64,
        ru_oublock: i64,
        ru_msgsnd: i64,
        ru_msgrcv: i64,
        ru_nsignals: i64,
        ru_nvcsw: i64,
        ru_nivcsw: i64,
    }
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    // SAFETY: `getrusage` is a pure read that fills the caller's buffer. The
    // struct is zero-initialised first, matches the platform's `rusage`
    // layout, and a non-zero return is treated as failure rather than trusted.
    unsafe {
        let mut u: Rusage = std::mem::zeroed();
        if getrusage(0 /* RUSAGE_SELF */, &mut u) != 0 {
            return None;
        }
        let hwm_kb = (u.ru_maxrss as u64) / 1024;
        Some((hwm_kb, hwm_kb))
    }
}

/// Every other target, wasm included: no memory figures available.
///
/// Phase markers still emit, so a trace captured in a browser keeps the same
/// shape as one from a phone — only the memory fields are absent.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn sample_rss_hwm_kb() -> Option<(u64, u64)> {
    None
}

/// Emit a phase marker on the `midnight_bench` target.
///
/// Consumed downstream by the wallet's log layer and its live benchmark stage
/// display. Reports MiB rather than KiB because the interesting magnitudes are
/// hundreds of MiB and the extra precision is noise.
pub(crate) fn log_phase(name: &'static str) {
    if let Some((rss_kb, hwm_kb)) = sample_rss_hwm_kb() {
        tracing::info!(
            target: "midnight_bench",
            stage = name,
            rss_mb = rss_kb / 1024,
            hwm_mb = hwm_kb / 1024,
        );
    } else {
        tracing::info!(target: "midnight_bench", stage = name);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sampler_is_callable_on_every_target() {
        // The point is that this links and runs at all: the macOS arm is FFI
        // and the Linux arm touches the filesystem, so a target-gating mistake
        // shows up as a link or compile failure rather than a wrong value.
        let sampled = sample_rss_hwm_kb();
        #[cfg(any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos",
            target_os = "ios"
        ))]
        {
            let (rss, hwm) = sampled.expect("a supported platform must report figures");
            assert!(
                rss > 0,
                "resident set size should be non-zero in a live process"
            );
            assert!(hwm >= rss, "high-water mark cannot be below current RSS");
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos",
            target_os = "ios"
        )))]
        assert!(sampled.is_none(), "unsupported targets report nothing");
    }

    #[test]
    fn log_phase_does_not_panic_without_a_subscriber() {
        // Called from the prover's hot path, where no tracing subscriber is
        // guaranteed to be installed.
        log_phase("test.phase");
    }
}
