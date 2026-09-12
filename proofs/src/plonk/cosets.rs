//! Coset construction for the prover, with an optional disk-spilled backing.
//!
//! Always compiled. The spilled arm exists only under the `disk-spill` feature,
//! so a wasm consumer gets the plain heap path with no filesystem dependency
//! reachable and none shipped — while the prover's call sites stay free of
//! `cfg`, which is where scattered gating would otherwise accumulate.

#[cfg(feature = "disk-spill")]
use crate::plonk::mmap_pk::{MmappedPolys, spill_with_transform};
use crate::poly::{Coeff, ExtendedLagrangeCoeff, Polynomial};

/// Either heap-resident cosets or cosets spilled to a mapped tempfile.
///
/// Both arms hand out `&[Polynomial<F, ExtendedLagrangeCoeff>]`, which is what
/// `evaluate_numerator` already takes — so the choice is invisible to the
/// evaluator and no call site changes shape.
pub enum Cosets<F> {
    /// The default: built in parallel and held in memory.
    Heap(Vec<Polynomial<F, ExtendedLagrangeCoeff>>),
    /// Streamed to a tempfile and mapped back, one polynomial at a time.
    #[cfg(feature = "disk-spill")]
    Spilled(MmappedPolys<F, ExtendedLagrangeCoeff>),
}

impl<F> std::fmt::Debug for Cosets<F> {
    /// Deliberately does not print the values, matching `MmappedPolys`: for the
    /// spilled arm that would fault the whole file into RAM, defeating the
    /// point of having spilled it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cosets::Heap(v) => f.debug_struct("Cosets::Heap").field("n_polys", &v.len()).finish(),
            #[cfg(feature = "disk-spill")]
            Cosets::Spilled(m) => f.debug_struct("Cosets::Spilled").field("inner", m).finish(),
        }
    }
}

impl<F> Cosets<F> {
    /// Borrow as a flat slice, whichever arm this is.
    pub fn as_slice(&self) -> &[Polynomial<F, ExtendedLagrangeCoeff>] {
        match self {
            Cosets::Heap(v) => v,
            #[cfg(feature = "disk-spill")]
            Cosets::Spilled(m) => m.as_slice(),
        }
    }
}

/// Whether to spill cosets for a circuit of size `k`.
///
/// Off unless `MIDNIGHT_SPILL_COSETS` is `1`/`true`, and then only at or above
/// `MIDNIGHT_SPILL_FLOOR_K` (default 18). The floor exists because spilling is
/// a loss at small `k` — the file write and page faults cost more than the heap
/// the small cosets would have occupied. It earns its keep only once the cosets
/// approach the memory ceiling.
pub fn should_spill_cosets(k: u32) -> bool {
    const DEFAULT_SPILL_FLOOR_K: u32 = 18;
    let floor = std::env::var("MIDNIGHT_SPILL_FLOOR_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SPILL_FLOOR_K);
    let enabled = matches!(
        std::env::var("MIDNIGHT_SPILL_COSETS").as_deref(),
        Ok("1") | Ok("true")
    );
    spill_decision(k, enabled, floor)
}

/// The decision itself, separated from reading the environment.
///
/// Split out so it can be tested directly: mutating process environment from a
/// test is `unsafe` under Rust 2024 and races with every other test in the
/// binary, so the gate would otherwise be either untested or flaky.
fn spill_decision(k: u32, enabled: bool, floor: u32) -> bool {
    enabled && k >= floor
}

/// Build extended-domain cosets from `polys`, spilling to disk when
/// [`should_spill_cosets`] says so and the spill succeeds.
///
/// A spill failure is **not** fatal: it falls back to the heap path and the
/// proof is still produced. Running out of tempfile space should degrade to the
/// behaviour we had before this optimisation existed, not abort a proof.
pub fn build_cosets<F, D>(polys: &[Polynomial<F, Coeff>], k: u32, to_extended: D) -> Cosets<F>
where
    F: Send + Sync,
    D: Fn(&Polynomial<F, Coeff>) -> Polynomial<F, ExtendedLagrangeCoeff> + Send + Sync,
{
    use rayon::prelude::*;

    #[cfg(feature = "disk-spill")]
    if should_spill_cosets(k) && !polys.is_empty() {
        let n_per = to_extended(&polys[0]).values.len();
        if let Ok(m) = spill_with_transform(polys, n_per, &to_extended) {
            return Cosets::Spilled(m);
        }
    }
    // Without `disk-spill` the gate is dead weight; keep the parameter so the
    // signature does not change with the feature.
    #[cfg(not(feature = "disk-spill"))]
    let _ = (k, should_spill_cosets(k));
    // The heap arm stays parallel. Spilling is sequential by construction —
    // streaming one polynomial at a time is what keeps peak heap at ~1 poly —
    // so the two arms trade throughput against memory, and the default path
    // must not quietly lose the parallelism it had before this existed.
    Cosets::Heap(polys.par_iter().map(&to_extended).collect())
}

#[cfg(test)]
mod test {
    #[cfg(feature = "disk-spill")]
    use midnight_curves::Fq as Fp;

    use super::*;
    #[cfg(feature = "disk-spill")]
    use crate::plonk::mmap_pk::spill_with_transform;

    #[test]
    #[cfg(feature = "disk-spill")]
    fn spilled_and_heap_cosets_agree() {
        // The property that matters: turning the optimisation on must not
        // change the answer. A spill producing different cosets would still
        // pass the rest of the suite, because nothing else compares the arms.
        let polys: Vec<Polynomial<Fp, Coeff>> = (0..3)
            .map(|i| Polynomial {
                values: (0..8).map(|j| Fp::from((i * 8 + j) as u64)).collect(),
                _marker: std::marker::PhantomData,
            })
            .collect();
        let widen = |p: &Polynomial<Fp, Coeff>| Polynomial::<Fp, ExtendedLagrangeCoeff> {
            values: p.values.clone(),
            _marker: std::marker::PhantomData,
        };

        // Both arms are constructed directly. Going through `build_cosets`
        // would make the test depend on process-wide environment another test
        // - or the caller's shell - may have set, which is exactly what
        // splitting `spill_decision` out was meant to avoid.
        let heap = Cosets::Heap(polys.iter().map(widen).collect());
        let n_per = widen(&polys[0]).values.len();
        let spilled = Cosets::Spilled(spill_with_transform(&polys, n_per, widen).unwrap());

        assert_eq!(heap.as_slice().len(), spilled.as_slice().len());
        for (h, sp) in heap.as_slice().iter().zip(spilled.as_slice()) {
            assert_eq!(h.values, sp.values, "spilled cosets must equal heap cosets");
        }
    }

    #[test]
    fn spill_gate_needs_both_the_switch_and_the_floor() {
        // Guards the gate itself: the difference between an optimisation that
        // engages where it helps and one that fires at every k.
        assert!(
            !spill_decision(17, true, 18),
            "below the floor must not spill"
        );
        assert!(spill_decision(18, true, 18), "at the floor must spill");
        assert!(spill_decision(20, true, 18), "above the floor must spill");
        assert!(
            !spill_decision(20, false, 18),
            "switch off must not spill at any k"
        );
        assert!(!spill_decision(0, false, 0), "both off is off");
    }
}
