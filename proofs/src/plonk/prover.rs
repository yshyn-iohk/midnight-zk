use std::{
    collections::{BTreeSet, HashMap, HashSet},
    hash::Hash,
    iter,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::RangeTo,
    sync::Arc,
};

use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use rand_core::{CryptoRng, RngCore};

use super::{
    circuit::{
        sealed::{self},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    lookup, permutation, vanishing, Error, ProvingKey,
};
#[cfg(feature = "committed-instances")]
use crate::poly::EvaluationDomain;
use crate::{
    circuit::Value,
    plonk::{traces::ProverTrace, trash},
    poly::{
        batch_invert_rational, commitment::PolynomialCommitmentScheme, Coeff,
        ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, PolynomialRepresentation, ProverQuery,
    },
    transcript::{Hashable, Sampleable, Transcript},
    utils::{arithmetic::eval_polynomial, rational::Rational},
};

#[cfg(feature = "committed-instances")]
/// Commit to a vector of raw instances. This function can be used to prepare
/// the committed instances that the verifier will be provided with when this
/// feature is enabled.
pub fn commit_to_instances<F, CS: PolynomialCommitmentScheme<F>>(
    params: &CS::Parameters,
    domain: &EvaluationDomain<F>,
    instances: &[F],
) -> CS::Commitment
where
    F: WithSmallOrderMulGroup<3> + Ord + FromUniformBytes<64>,
{
    let mut poly = domain.empty_lagrange();
    for (poly_eval, value) in poly.iter_mut().zip(instances.iter()) {
        *poly_eval = *value;
    }
    CS::commit_lagrange(params, &poly)
}

/// This computes a proof trace for the provided `circuits` when given the
/// public parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
///
/// The trace can then be used to finalise proofs, or to fold them.
pub(crate) fn compute_trace<
    F,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
    ConcreteCircuit: Circuit<F>,
>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    circuits: &[ConcreteCircuit],
    // The prover needs to get all instances in non-committed form. However,
    // the first `nb_committed_instances` instance columns are dedicated for
    // instances that the verifier receives in committed form.
    #[cfg(feature = "committed-instances")] nb_committed_instances: usize,
    instances: &[&[&[F]]],
    mut rng: impl RngCore + CryptoRng,
    transcript: &mut T,
) -> Result<ProverTrace<F>, Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    #[cfg(not(feature = "committed-instances"))]
    let nb_committed_instances: usize = 0;

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instances in instances.iter() {
        if instances.len() != pk.vk.cs.num_instance_columns
            || instances.len() < nb_committed_instances
        {
            return Err(Error::InvalidInstances);
        }
    }

    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;

    let domain = &pk.vk.domain;

    log_phase("trace.compute_instances.start");
    let instance = compute_instances(params, pk, instances, nb_committed_instances, transcript)?;
    log_phase("trace.compute_instances.end");

    log_phase("trace.parse_advices.start");
    let (advice, challenges) =
        parse_advices(params, pk, circuits, instances, transcript, &mut rng)?;
    log_phase("trace.parse_advices.end");

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: F = transcript.squeeze_challenge();

    log_phase("trace.lookups_permuted.start");
    let lookups: Vec<Vec<lookup::prover::Permuted<F>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    log_phase("trace.lookups_permuted.end");

    // Sample beta challenge
    let beta: F = transcript.squeeze_challenge();

    // Sample gamma challenge
    let gamma: F = transcript.squeeze_challenge();

    log_phase("trace.permutations_commit.start");
    // Commit to permutations.
    let permutations: Vec<permutation::prover::Committed<F>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    log_phase("trace.permutations_commit.end");

    log_phase("trace.lookups_product.start");
    let lookups: Vec<Vec<lookup::prover::Committed<F>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log_phase("trace.lookups_product.end");

    // Trash argument
    let trash_challenge: F = transcript.squeeze_challenge();

    let trashcans: Vec<Vec<trash::prover::Committed<F>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            pk.vk
                .cs
                .trashcans
                .iter()
                .map(|trash| {
                    trash.commit::<CS, _>(
                        params,
                        domain,
                        trash_challenge,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::<F, CS>::commit(params, domain, &mut rng, transcript)?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: F = transcript.squeeze_challenge();

    let (instance_polys, instance_values) =
        instance.into_iter().map(|i| (i.instance_polys, i.instance_values)).unzip();

    let advice_polys = advice
        .into_iter()
        .map(|a| {
            a.advice_polys
                .into_iter()
                .map(|p| domain.lagrange_to_coeff(p))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(ProverTrace {
        advice_polys,
        instance_polys,
        instance_values,
        vanishing,
        lookups,
        trashcans,
        permutations,
        challenges,
        beta,
        gamma,
        theta,
        trash_challenge,
        y,
    })
}

/// This takes the computed trace of a set of witnesses and creates a proof
/// for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
/// Sample `(VmRSS, VmHWM)` in KiB from `/proc/self/status`. Returns
/// `None` on non-Linux/Android targets. Used by [`log_phase`] to
/// emit per-phase memory snapshots through the `midnight_bench`
/// tracing target — the dioxus-wallet `BenchStageLayer` captures
/// these and renders them in the Benchmark tab stage pill, plus
/// they appear as ordinary entries in the Logs tab.
///
/// Reading `/proc/self/status` is cheap (single syscall, ~few KiB
/// of kernel text), so we can call this freely at phase boundaries
/// without measurable wall-clock overhead.
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

/// macOS / iOS path. There's no `/proc`; use the libc-ish
/// `mach_task_basic_info` via `getrusage(RUSAGE_SELF)`. Same
/// `(rss_kb, peak_kb)` return shape so callers don't branch.
/// We can't read crate-level deps cleanly here, so call libc
/// directly via the link-shim that ships with the Rust runtime.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sample_rss_hwm_kb() -> Option<(u64, u64)> {
    // SAFETY: `getrusage` is async-signal-safe and pure-read; the
    // `rusage` struct is zero-init then filled by the kernel.
    #[allow(unsafe_code)]
    unsafe {
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        #[repr(C)]
        struct Timeval { tv_sec: i64, tv_usec: i32 }
        #[repr(C)]
        struct Rusage {
            ru_utime: Timeval,
            ru_stime: Timeval,
            ru_maxrss: i64,        // bytes on macOS
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
        let mut u: Rusage = std::mem::zeroed();
        if getrusage(0 /* RUSAGE_SELF */, &mut u) != 0 {
            return None;
        }
        // macOS reports `ru_maxrss` in bytes. Convert to KiB to
        // match the Linux semantics. We don't have a true "current
        // RSS" — `ru_maxrss` is the high-water mark — so report it
        // as both rss and hwm; callers see them tracking the same.
        let hwm_kb = (u.ru_maxrss as u64) / 1024;
        Some((hwm_kb, hwm_kb))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios")))]
fn sample_rss_hwm_kb() -> Option<(u64, u64)> {
    None
}

/// Emit a phase-marker tracing event. Captured by both
/// `WalletLogLayer` (→ Logs tab + redb persistence) and
/// `BenchStageLayer` (→ live stage pill on the Benchmark tab).
/// `rss_mb` is the live resident set size at the moment of the
/// call; `hwm_mb` is the high-water mark across the process
/// lifetime so far. Together they show which phase is responsible
/// for each step-up in peak memory.
/// Holder for cosets that were built one-by-one, written to a
/// tempfile, then mmap'd. The `polys` vec contains `Polynomial<F>`
/// instances whose `values: Vec<F>` are *non-owning* views into
/// the mmap region — they MUST NOT be dropped normally (a Vec drop
/// would call the allocator's `free` on a pointer the allocator
/// doesn't own). We wrap them in `ManuallyDrop` to suppress the
/// destructor and let the `Arc<Mmap>` clean up the actual storage.
///
/// `as_slice` returns `&[Polynomial<F, ExtendedLagrangeCoeff>]` —
/// the layout-transparent transmute is sound because
/// `ManuallyDrop<T>` is `#[repr(transparent)]` over T, and
/// `evaluate_h` only reads the slice (never mutates / drops
/// elements through `&[T]`).
///
/// The tempfile is deleted on drop via the `TempPath` it holds.
pub(super) struct SpilledCosets<F> {
    _mmap: Arc<memmap2::Mmap>,
    polys: Vec<ManuallyDrop<Polynomial<F, ExtendedLagrangeCoeff>>>,
    _tmp_path: tempfile::TempPath,
}

impl<F> SpilledCosets<F> {
    pub(super) fn as_slice(&self) -> &[Polynomial<F, ExtendedLagrangeCoeff>] {
        // SAFETY: ManuallyDrop<T> is #[repr(transparent)]; layout
        // is identical to T. We expose only `&[T]` (immutable
        // shared access); no destructors or moves occur through
        // this slice. The underlying mmap is kept alive by `_mmap`.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(
                self.polys.as_ptr() as *const Polynomial<F, ExtendedLagrangeCoeff>,
                self.polys.len(),
            )
        }
    }
}

/// Build cosets one at a time, write each to `tmp`, drop before
/// the next is built. After all are written, mmap the file and
/// produce `Polynomial<F>` views over the mapped pages.
///
/// Peak heap during the build is one coset (~`4n × size_of(F)`
/// bytes) plus the destination Vec being assembled — at k=20 with
/// 50 fixed columns, the per-column transient is ~128 MiB whereas
/// the previous lazy `.collect()` would have held all 50 ×
/// 128 MiB = 6.4 GiB at once.
pub(super) fn spill_cosets_to_disk<F>(
    polys: &[Polynomial<F, Coeff>],
    domain: &crate::poly::EvaluationDomain<F>,
) -> std::io::Result<SpilledCosets<F>>
where
    F: WithSmallOrderMulGroup<3>,
{
    use std::io::Write as _;

    let coset_size = polys
        .first()
        .map(|_| 1usize << domain.extended_k())
        .unwrap_or(0);
    let elem_size = std::mem::size_of::<F>();

    // Allow overriding the spill directory via `MIDNIGHT_SPILL_DIR`.
    // Useful on devices where the default `TMPDIR` partition is too
    // small (e.g. Android emulator's `/data/local/tmp`, which is
    // often <1 GiB and may already be filled with pushed SRS files).
    // The directory must exist and be writable; falls back to the
    // platform default if the env var is unset or empty.
    let mut builder = tempfile::Builder::new();
    builder.prefix("midnight-cosets-");
    let mut tmp = match std::env::var("MIDNIGHT_SPILL_DIR") {
        Ok(dir) if !dir.is_empty() => builder.tempfile_in(dir)?,
        _ => builder.tempfile()?,
    };
    {
        let mut writer = std::io::BufWriter::new(&mut tmp);
        for p in polys.iter() {
            let coset = domain.coeff_to_extended(p.clone());
            // SAFETY: F is Copy + repr(transparent) over a fixed-size
            // integer array (BLS scalar = [u64; 4]); the in-memory
            // representation is stable and serialisable as raw bytes.
            // `coset.values` is a live Vec<F> we built this iteration.
            #[allow(unsafe_code)]
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    coset.values.as_ptr() as *const u8,
                    coset.values.len() * elem_size,
                )
            };
            writer.write_all(bytes)?;
            // `coset` drops here, freeing ~`4n × size_of(F)` bytes
            // before the next iteration allocates.
        }
        writer.flush()?;
    }
    let (file, tmp_path) = tmp.into_parts();
    // SAFETY: read-only mmap of a file we just wrote and own.
    #[allow(unsafe_code)]
    let mmap = unsafe { memmap2::Mmap::map(&file) }?;
    let mmap_arc = Arc::new(mmap);

    // Build non-owning Polynomial views into the mmap region.
    let n_polys = polys.len();
    let mut polys_view = Vec::with_capacity(n_polys);
    let base_ptr = mmap_arc.as_ptr() as *const F;
    for i in 0..n_polys {
        // SAFETY: the file holds exactly n_polys consecutive blocks
        // of `coset_size × elem_size` bytes, written above. The
        // returned `Vec<F>` is wrapped in ManuallyDrop so its
        // allocator-aware destructor never runs; the mmap region
        // is freed when `_mmap` (the Arc) drops at end of scope.
        // The pointer is `add(i * coset_size)` so each Polynomial
        // gets a disjoint, valid byte range.
        #[allow(unsafe_code)]
        let ptr = unsafe { base_ptr.add(i * coset_size) as *mut F };
        // SAFETY: `Vec::from_raw_parts` with cap == len means the
        // Vec believes it owns exactly `coset_size` elements at
        // `ptr`. We immediately wrap in ManuallyDrop so the Vec's
        // destructor (which would call `free(ptr)`) never runs.
        #[allow(unsafe_code)]
        let values = unsafe { Vec::from_raw_parts(ptr, coset_size, coset_size) };
        let poly = Polynomial::<F, ExtendedLagrangeCoeff> {
            values,
            _marker: PhantomData,
        };
        polys_view.push(ManuallyDrop::new(poly));
    }

    Ok(SpilledCosets {
        _mmap: mmap_arc,
        polys: polys_view,
        _tmp_path: tmp_path,
    })
}

fn log_phase(name: &'static str) {
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

/// `log_phase` exposed for `keygen.rs` (sibling module under `plonk`)
/// without changing visibility on the underlying helpers. Same
/// behaviour; same low-cost `/proc/self/status` read.
#[doc(hidden)]
pub(crate) fn log_phase_pub(name: &'static str) {
    log_phase(name)
}

pub(crate) fn finalise_proof<'a, F, CS: PolynomialCommitmentScheme<F>, T: Transcript>(
    params: &'a CS::Parameters,
    pk: &'a ProvingKey<F, CS>,
    // The prover needs to get all instances in non-committed form. However,
    // the first `nb_committed_instances` instance columns are dedicated for
    // instances that the verifier receives in committed form.
    #[cfg(feature = "committed-instances")] nb_committed_instances: usize,
    trace: ProverTrace<F>,
    transcript: &mut T,
) -> Result<(), Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    #[cfg(not(feature = "committed-instances"))]
    let nb_committed_instances: usize = 0;

    let domain = pk.get_vk().get_domain();

    log_phase("finalise.compute_h_poly.start");
    let h_poly = compute_h_poly(pk, &trace);
    log_phase("finalise.compute_h_poly.end");

    let ProverTrace {
        advice_polys,
        instance_polys,
        lookups,
        trashcans,
        permutations,
        vanishing,
        ..
    } = trace;

    // Construct the vanishing argument's h(X) commitments
    log_phase("finalise.vanishing_construct.start");
    let vanishing = vanishing.construct::<CS, T>(params, domain, h_poly, transcript)?;
    log_phase("finalise.vanishing_construct.end");

    let x: F = transcript.squeeze_challenge();

    write_evals_to_transcript(
        pk,
        nb_committed_instances,
        &instance_polys,
        &advice_polys,
        x,
        transcript,
    )?;

    let vanishing = vanishing.evaluate(x, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations: Vec<permutation::prover::Evaluated<F>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the lookups, if any, at omega^i x.
    let lookups: Vec<Vec<lookup::prover::Evaluated<F>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Evaluate the trashcans, if any, at x.
    let trashcans: Vec<Vec<trash::prover::Evaluated<F>>> = trashcans
        .into_iter()
        .map(|trash| -> Result<Vec<_>, _> {
            trash
                .into_iter()
                .map(|p| p.evaluate(x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    log_phase("finalise.compute_queries.start");
    let queries = compute_queries(
        pk,
        nb_committed_instances,
        &instance_polys,
        &advice_polys,
        &permutations,
        &lookups,
        &trashcans,
        &vanishing,
        x,
    );
    log_phase("finalise.multi_open.start");
    let res =
        CS::multi_open(params, &queries, transcript).map_err(|_| Error::ConstraintSystemFailure);
    log_phase("finalise.multi_open.end");
    res
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
//
// NOTE: Any change here must be mirrored in src/plonk/bench/prover.rs
// to ensure the benchmarks remain aligned with the real prover.
pub fn create_proof<
    F,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
    ConcreteCircuit: Circuit<F>,
>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    circuits: &[ConcreteCircuit],
    #[cfg(feature = "committed-instances")] nb_committed_instances: usize,
    instances: &[&[&[F]]],
    rng: impl RngCore + CryptoRng,
    transcript: &mut T,
) -> Result<(), Error>
where
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    log_phase("create_proof.compute_trace.start");
    let trace = compute_trace(
        params,
        pk,
        circuits,
        #[cfg(feature = "committed-instances")]
        nb_committed_instances,
        instances,
        rng,
        transcript,
    )?;
    log_phase("create_proof.compute_trace.end");
    log_phase("create_proof.finalise_proof.start");
    let res = finalise_proof(
        params,
        pk,
        #[cfg(feature = "committed-instances")]
        nb_committed_instances,
        trace,
        transcript,
    );
    log_phase("create_proof.finalise_proof.end");
    res
}

pub(super) fn compute_instances<F, CS, T>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    instances: &[&[&[F]]],
    nb_committed_instances: usize,
    transcript: &mut T,
) -> Result<Vec<InstanceSingle<F>>, Error>
where
    T: Transcript,
    CS: PolynomialCommitmentScheme<F>,
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<F>, Error> {
            let instance_values = instance
                .iter()
                .enumerate()
                .map(|(i, values)| {
                    // Committed instances go first.
                    let is_committed_instance = i < nb_committed_instances;
                    let mut poly = pk.vk.domain.empty_lagrange();
                    assert_eq!(poly.len(), pk.vk.domain.n as usize);
                    if values.len() > (poly.len() - (pk.vk.cs.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    if !is_committed_instance {
                        transcript.common(&F::from_u128(values.len() as u128))?;
                    }

                    for (poly_eval, value) in poly.iter_mut().zip(values.iter()) {
                        if !is_committed_instance {
                            transcript.common(value)?;
                        }
                        *poly_eval = *value;
                    }

                    if is_committed_instance {
                        transcript.common(&CS::commit_lagrange(params, &poly))?;
                    }

                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = pk.vk.domain.lagrange_from_vec(poly.to_vec());
                    pk.vk.domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

#[allow(clippy::type_complexity)]
pub(super) fn parse_advices<F, CS, ConcreteCircuit, T>(
    params: &CS::Parameters,
    pk: &ProvingKey<F, CS>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[F]]],
    transcript: &mut T,
    mut rng: impl RngCore + CryptoRng,
) -> Result<(Vec<AdviceSingle<F, LagrangeCoeff>>, Vec<F>), Error>
where
    F: WithSmallOrderMulGroup<3> + Sampleable<T::Hash>,
    CS: PolynomialCommitmentScheme<F>,
    ConcreteCircuit: Circuit<F>,
    T: Transcript,
    CS::Commitment: Hashable<T::Hash>,
    F: WithSmallOrderMulGroup<3>
        + Sampleable<T::Hash>
        + Hashable<T::Hash>
        + Hash
        + Ord
        + FromUniformBytes<64>,
{
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    let domain = &pk.vk.domain;
    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    let mut advice = vec![
        AdviceSingle::<F, LagrangeCoeff> {
            advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
        };
        instances.len()
    ];
    let mut challenges = HashMap::<usize, F>::with_capacity(meta.num_challenges);

    let unusable_rows_start = domain.n as usize - (meta.blinding_factors() + 1);
    for current_phase in pk.vk.cs.phases() {
        let column_indices = meta
            .advice_column_phase
            .iter()
            .enumerate()
            .filter_map(|(column_index, phase)| {
                if current_phase == *phase {
                    Some(column_index)
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();

        for ((circuit, advice), instances) in circuits.iter().zip(advice.iter_mut()).zip(instances)
        {
            let mut witness = WitnessCollection {
                k: domain.k(),
                current_phase,
                advice: vec![domain.empty_lagrange_rational(); meta.num_advice_columns],
                unblinded_advice: HashSet::from_iter(meta.unblinded_advice_columns.clone()),
                instances,
                challenges: &challenges,
                // The prover will not be allowed to assign values to advice
                // cells that exist within inactive rows, which include some
                // number of blinding factors and an extra row for use in the
                // permutation argument.
                usable_rows: ..unusable_rows_start,
                _marker: std::marker::PhantomData,
            };

            // Synthesize the circuit to obtain the witness and other information.
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )?;

            let mut advice_values = batch_invert_rational::<F>(
                witness
                    .advice
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            Some(advice)
                        } else {
                            None
                        }
                    })
                    .collect(),
            );

            for (column_index, advice_values) in column_indices.iter().zip(&mut advice_values) {
                if !witness.unblinded_advice.contains(column_index) {
                    for cell in &mut advice_values[unusable_rows_start..] {
                        *cell = F::random(&mut rng);
                    }
                } else {
                    #[cfg(debug_assertions)]
                    for cell in &advice_values[unusable_rows_start..] {
                        assert_eq!(*cell, F::ZERO);
                    }
                }
            }

            let advice_commitments: Vec<_> =
                advice_values.iter().map(|poly| CS::commit_lagrange(params, poly)).collect();

            for commitment in &advice_commitments {
                transcript.write(commitment)?;
            }
            for (column_index, advice_values) in column_indices.iter().zip(advice_values) {
                advice.advice_polys[*column_index] = advice_values;
            }
        }

        for (index, phase) in meta.challenge_phase.iter().enumerate() {
            if current_phase == *phase {
                let existing = challenges.insert(index, transcript.squeeze_challenge());
                assert!(existing.is_none());
            }
        }
    }

    assert_eq!(challenges.len(), meta.num_challenges);
    let challenges = (0..meta.num_challenges)
        .map(|index| challenges.remove(&index).unwrap())
        .collect::<Vec<_>>();

    Ok((advice, challenges))
}

pub(super) fn compute_h_poly<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>(
    pk: &ProvingKey<F, CS>,
    trace: &ProverTrace<F>,
) -> Polynomial<F, ExtendedLagrangeCoeff> {
    let ProverTrace {
        advice_polys,
        instance_polys,
        lookups,
        trashcans,
        permutations,
        challenges,
        beta,
        gamma,
        theta,
        trash_challenge,
        y,
        ..
    } = &trace;

    // ── streaming-iteration-3 S1 ──
    // The advice + instance per-instance coset arrays are the single
    // biggest unspilled allocation in `finalise_proof`. At k=21 with
    // ~30 advice cols + extended_factor=8 the eager Vec<Vec<Polynomial>>
    // is ~7.7 GiB coresident — directly responsible for the bulk of the
    // pre-S1 phys_footprint we measured (8.3 GiB iOS sim, commit
    // 7b6f7a5). Same env-var gating as fixed/perm cosets:
    // `MIDNIGHT_SPILL_COSETS=1` + `MIDNIGHT_SPILL_FLOOR_K` (default 18).
    //
    // Spill path: each instance's advice (or instance) polys go through
    // `spill_cosets_to_disk` → a `SpilledCosets<F>` that holds an mmap
    // arc and `Polynomial` views into the mapped pages. Drop of the
    // whole `Vec<CosetsForInstance>` cleans the tempfile up.
    //
    // In-memory path stays the default for hosts with abundant RAM
    // (and for low-k where the spill IO overhead isn't worth it).
    const DEFAULT_SPILL_FLOOR_K: u32 = 18;
    let spill_floor_k: u32 = std::env::var("MIDNIGHT_SPILL_FLOOR_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SPILL_FLOOR_K);
    let want_spill_advice = matches!(
        std::env::var("MIDNIGHT_SPILL_COSETS").as_deref(),
        Ok("1") | Ok("true")
    ) && pk.vk.domain.k() >= spill_floor_k;

    enum CosetsForInstance<F: WithSmallOrderMulGroup<3>> {
        InMem(Vec<Polynomial<F, ExtendedLagrangeCoeff>>),
        Spilled(SpilledCosets<F>),
    }
    impl<F: WithSmallOrderMulGroup<3>> CosetsForInstance<F> {
        fn as_slice(&self) -> &[Polynomial<F, ExtendedLagrangeCoeff>] {
            match self {
                Self::InMem(v) => v.as_slice(),
                Self::Spilled(s) => s.as_slice(),
            }
        }
    }

    // Calculate the advice and instance cosets.
    let advice_cosets: Vec<CosetsForInstance<F>> = advice_polys
        .iter()
        .map(|advice_polys| {
            if want_spill_advice {
                log_phase("finalise.compute_h_poly.spill_advice_cosets.start");
                let s = spill_cosets_to_disk(advice_polys, &pk.vk.domain)
                    .expect("spill advice_cosets to tempfile");
                log_phase("finalise.compute_h_poly.spill_advice_cosets.end");
                CosetsForInstance::Spilled(s)
            } else {
                CosetsForInstance::InMem(
                    advice_polys
                        .iter()
                        .map(|poly| pk.vk.get_domain().coeff_to_extended(poly.clone()))
                        .collect(),
                )
            }
        })
        .collect();
    let instance_cosets: Vec<CosetsForInstance<F>> = instance_polys
        .iter()
        .map(|instance_polys| {
            if want_spill_advice {
                log_phase("finalise.compute_h_poly.spill_instance_cosets.start");
                let s = spill_cosets_to_disk(instance_polys, &pk.vk.domain)
                    .expect("spill instance_cosets to tempfile");
                log_phase("finalise.compute_h_poly.spill_instance_cosets.end");
                CosetsForInstance::Spilled(s)
            } else {
                CosetsForInstance::InMem(
                    instance_polys
                        .iter()
                        .map(|poly| pk.vk.get_domain().coeff_to_extended(poly.clone()))
                        .collect(),
                )
            }
        })
        .collect();

    // Materialise the fixed + permutation cosets right before
    // `evaluate_h`. Two paths:
    //
    // - `MIDNIGHT_SPILL_COSETS=1` — disk-backed: build each coset
    //   one at a time, write to a tempfile, drop before the next.
    //   Mmap the result. Peak in-memory transient per column.
    //   Required for k ≥ 20 on phones where the full
    //   `Vec<Polynomial>` (~6+ GiB at k=20) can't fit alongside
    //   the existing prove working set.
    //
    // - default — in-memory `.collect()`: faster (~1 disk write
    //   pass saved) but holds all cosets coresident. Fine through
    //   k=19 on mobile, dies at k=20.
    //
    // Forward-compat: if the ProvingKey carries a pre-built
    // cached `fixed_cosets` / `permutation.cosets` (from an older
    // eager-keygen path or a deserialised PK), use it as-is.
    // Only spill at high `k`. At small `k` the entire coset
    // collection fits comfortably in heap; round-tripping through
    // a tempfile is pure overhead (measured: +46–47 % prove
    // time at k=16/17 on a Samsung S24 Ultra when the wallet
    // unconditionally set `MIDNIGHT_SPILL_COSETS=1`). The
    // mobile wallet sets the env var at process start regardless
    // of `k`; the cheap fix is to gate the spill on `k` here so
    // small-k proves keep their in-memory fast path.
    //
    // Override the floor with `MIDNIGHT_SPILL_FLOOR_K`. Set to
    // `0` to spill at every `k` (useful for testing the spill
    // path directly).
    //
    // S1 (iteration 3): the spill_floor_k / want_spill computation
    // also drives the advice/instance cosets spill earlier in this
    // function. Reuse `want_spill_advice` here as the shared spill
    // gate so all four coset categories (fixed/perm/advice/instance)
    // follow the same env-var contract.
    let want_spill = want_spill_advice;

    let computed_fixed_cosets;
    let spilled_fixed_cosets;
    let fixed_cosets_ref: &[crate::poly::Polynomial<F, ExtendedLagrangeCoeff>] =
        if !pk.fixed_cosets.is_empty() {
            &pk.fixed_cosets
        } else if want_spill {
            log_phase("finalise.compute_h_poly.spill_fixed_cosets.start");
            spilled_fixed_cosets = spill_cosets_to_disk(&pk.fixed_polys, &pk.vk.domain)
                .expect("spill fixed_cosets to tempfile");
            log_phase("finalise.compute_h_poly.spill_fixed_cosets.end");
            spilled_fixed_cosets.as_slice()
        } else {
            log_phase("finalise.compute_h_poly.materialise_fixed_cosets.start");
            computed_fixed_cosets = pk
                .fixed_polys
                .iter()
                .map(|p| pk.vk.domain.coeff_to_extended(p.clone()))
                .collect::<Vec<_>>();
            log_phase("finalise.compute_h_poly.materialise_fixed_cosets.end");
            &computed_fixed_cosets
        };

    let computed_perm_cosets;
    let spilled_perm_cosets;
    let perm_cosets_ref: &[crate::poly::Polynomial<F, ExtendedLagrangeCoeff>] =
        if !pk.permutation.cosets.is_empty() {
            &pk.permutation.cosets
        } else if want_spill {
            log_phase("finalise.compute_h_poly.spill_perm_cosets.start");
            spilled_perm_cosets =
                spill_cosets_to_disk(&pk.permutation.polys, &pk.vk.domain)
                    .expect("spill permutation cosets to tempfile");
            log_phase("finalise.compute_h_poly.spill_perm_cosets.end");
            spilled_perm_cosets.as_slice()
        } else {
            log_phase("finalise.compute_h_poly.materialise_perm_cosets.start");
            computed_perm_cosets = pk
                .permutation
                .polys
                .iter()
                .map(|p| pk.vk.domain.coeff_to_extended(p.clone()))
                .collect::<Vec<_>>();
            log_phase("finalise.compute_h_poly.materialise_perm_cosets.end");
            &computed_perm_cosets
        };

    // Evaluate the h(X) polynomial
    let h_poly = pk.ev.evaluate_h::<ExtendedLagrangeCoeff>(
        &pk.vk.domain,
        &pk.vk.cs,
        &advice_cosets.iter().map(|a| a.as_slice()).collect::<Vec<_>>(),
        &instance_cosets.iter().map(|i| i.as_slice()).collect::<Vec<_>>(),
        fixed_cosets_ref,
        challenges,
        *y,
        *beta,
        *gamma,
        *theta,
        *trash_challenge,
        lookups,
        trashcans,
        permutations,
        &pk.l0,
        &pk.l_last,
        &pk.l_active_row,
        perm_cosets_ref,
    );
    // `computed_fixed_cosets` / `computed_perm_cosets` (if we built
    // them) drop here — releasing both extended-domain expansions
    // before vanishing.construct and multi_open run.
    log_phase("finalise.compute_h_poly.drop_cosets.end");
    h_poly
}

pub(super) fn write_evals_to_transcript<F, CS, T>(
    pk: &ProvingKey<F, CS>,
    nb_committed_instances: usize,
    instance_polys: &[Vec<Polynomial<F, Coeff>>],
    advice_polys: &[Vec<Polynomial<F, Coeff>>],
    x: F,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: WithSmallOrderMulGroup<3> + Hashable<T::Hash>,
    CS: PolynomialCommitmentScheme<F>,
    T: Transcript,
{
    let domain = &pk.vk.domain;
    let meta = &pk.vk.cs;
    // Compute and hash evals for the polynomials of the committed instances of
    // each circuit
    for instance in instance_polys.iter() {
        // Evaluate polynomials at omega^i x
        for &(column, at) in meta.instance_queries.iter() {
            if column.index() < nb_committed_instances {
                let eval = eval_polynomial(&instance[column.index()], domain.rotate_omega(x, at));
                transcript.write(&eval)?;
            }
        }
    }

    // Compute and hash advice evals for each circuit instance
    for advice in advice_polys.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(&advice[column.index()], domain.rotate_omega(x, at))
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write(eval)?;
        }
    }

    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(x, at))
        })
        .collect();

    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write(eval)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_queries<
    'a,
    F: WithSmallOrderMulGroup<3>,
    CS: PolynomialCommitmentScheme<F>,
>(
    pk: &'a ProvingKey<F, CS>,
    nb_committed_instances: usize,
    instance_polys: &'a [Vec<Polynomial<F, Coeff>>],
    advice_polys: &'a [Vec<Polynomial<F, Coeff>>],
    permutations: &'a [permutation::prover::Evaluated<F>],
    lookups: &'a [Vec<lookup::prover::Evaluated<F>>],
    trashcans: &'a [Vec<trash::prover::Evaluated<F>>],
    vanishing: &'a vanishing::prover::Evaluated<F>,
    x: F,
) -> Vec<ProverQuery<'a, F>> {
    let domain = pk.vk.get_domain();
    instance_polys
        .iter()
        .zip(advice_polys.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .zip(trashcans.iter())
        .flat_map(
            move |((((instance, advice), permutation), lookups), trash)| {
                iter::empty()
                    .chain(
                        pk.vk.cs.instance_queries.iter().filter_map(move |&(column, at)| {
                            if column.index() < nb_committed_instances {
                                Some(ProverQuery {
                                    point: domain.rotate_omega(x, at),
                                    poly: &instance[column.index()],
                                })
                            } else {
                                None
                            }
                        }),
                    )
                    .chain(
                        pk.vk.cs.advice_queries.iter().map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(x, at),
                            poly: &advice[column.index()],
                        }),
                    )
                    .chain(permutation.open(pk, x))
                    .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                    .chain(trash.iter().flat_map(move |p| p.open(x)))
            },
        )
        .chain(
            pk.vk.cs.fixed_queries.iter().map(move |&(column, at)| ProverQuery {
                point: domain.rotate_omega(x, at),
                poly: &pk.fixed_polys[column.index()],
            }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x))
        .collect::<Vec<_>>()
}

#[derive(Clone)]
pub(super) struct InstanceSingle<F: PrimeField> {
    pub instance_values: Vec<Polynomial<F, LagrangeCoeff>>,
    pub instance_polys: Vec<Polynomial<F, Coeff>>,
}

#[derive(Clone)]
pub(super) struct AdviceSingle<F: PrimeField, B: PolynomialRepresentation> {
    pub advice_polys: Vec<Polynomial<F, B>>,
}

struct WitnessCollection<'a, F: Field> {
    k: u32,
    current_phase: sealed::Phase,
    advice: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
    unblinded_advice: HashSet<usize>,
    challenges: &'a HashMap<usize, F>,
    instances: &'a [&'a [F]],
    usable_rows: RangeTo<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> Assignment<F> for WitnessCollection<'_, F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.instances
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| Value::known(*v))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Advice>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Rational<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Ignore assignment of advice column in different phase than current one.
        if self.current_phase != column.column_type().phase {
            return Ok(());
        }

        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        *self
            .advice
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Fixed>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Rational<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) -> Result<(), Error> {
        // We only care about advice columns here

        Ok(())
    }

    fn fill_from_row(
        &mut self,
        _: Column<Fixed>,
        _: usize,
        _: Value<Rational<F>>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.challenges
            .get(&challenge.index())
            .cloned()
            .map(Value::known)
            .unwrap_or_else(Value::unknown)
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

#[test]
#[cfg(feature = "dev-curves")]
fn test_create_proof() {
    use midnight_curves::bn256::{Bn256, Fr};
    use rand_core::OsRng;

    use crate::{
        circuit::SimpleFloorPlanner,
        plonk::{keygen_pk, keygen_vk_with_k},
        poly::kzg::{params::ParamsKZG, KZGCommitmentScheme},
        transcript::CircuitTranscript,
    };

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    const K: u32 = 4;
    let params: ParamsKZG<Bn256> = ParamsKZG::unsafe_setup(K, OsRng);
    let vk = keygen_vk_with_k(&params, &MyCircuit, K).expect("keygen_vk should not fail");
    let pk = keygen_pk(vk, &MyCircuit).expect("keygen_pk should not fail");
    let mut transcript = CircuitTranscript::<_>::init();

    // Create proof with wrong number of instances
    let proof = create_proof::<Fr, KZGCommitmentScheme<Bn256>, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        #[cfg(feature = "committed-instances")]
        0,
        &[],
        OsRng,
        &mut transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    create_proof::<Fr, KZGCommitmentScheme<Bn256>, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        #[cfg(feature = "committed-instances")]
        0,
        &[&[], &[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
}
