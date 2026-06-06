use std::{any::TypeId, fmt::Debug};

use ff::Field;
use group::{Curve, Group};
use midnight_curves::{
    msm::msm_best,
    pairing::{Engine, MillerLoopResult, MultiMillerLoop},
    CurveAffine, Fq, G1Projective,
};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use super::params::ParamsVerifierKZG;
use crate::{
    poly::{
        commitment::{Guard, PolynomialCommitmentScheme},
        kzg::KZGCommitmentScheme,
        Error,
    },
    utils::{
        arithmetic::{CurveExt, MSM},
        helpers::ProcessedSerdeObject,
    },
};

/// A multiscalar multiplication in the polynomial commitment scheme
#[derive(Clone, Default, Debug)]
pub struct MSMKZG<E: Engine> {
    pub(crate) scalars: Vec<E::Fr>,
    pub(crate) bases: Vec<E::G1>,
}

impl<E: Engine> MSMKZG<E> {
    /// Create an empty MSM instance
    pub fn init() -> Self {
        MSMKZG {
            scalars: vec![],
            bases: vec![],
        }
    }

    /// Create an MSM from various MSMs
    pub fn from_many(msms: Vec<Self>) -> Self {
        let len = msms.iter().map(|m| m.scalars.len()).sum();

        let mut scalars = Vec::with_capacity(len);
        let mut bases = Vec::with_capacity(len);

        for mut msm in msms {
            scalars.append(&mut msm.scalars);
            bases.append(&mut msm.bases);
        }

        Self { scalars, bases }
    }

    /// Create a new MSM from a given base (with scalar of 1).
    pub fn from_base(base: &E::G1) -> Self {
        MSMKZG {
            scalars: vec![E::Fr::ONE],
            bases: vec![*base],
        }
    }
}

impl<E: Engine + Debug> MSM<E::G1Affine> for MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn append_term(&mut self, scalar: E::Fr, point: E::G1) {
        self.scalars.push(scalar);
        self.bases.push(point);
    }

    fn add_msm(&mut self, other: &Self) {
        self.scalars.reserve(other.scalars().len());
        self.scalars.extend_from_slice(&other.scalars());

        self.bases.reserve(other.bases().len());
        self.bases.extend_from_slice(&other.bases());
    }

    fn scale(&mut self, factor: E::Fr) {
        self.scalars.par_iter_mut().for_each(|s| {
            *s *= &factor;
        })
    }

    fn check(&self) -> bool {
        bool::from(self.eval().is_identity())
    }

    fn eval(&self) -> E::G1 {
        if self.scalars == vec![E::Fr::ONE] {
            self.bases[0]
        } else {
            msm_specific::<E::G1Affine>(&self.scalars, &self.bases)
        }
    }

    fn bases(&self) -> Vec<E::G1> {
        self.bases.clone()
    }

    fn scalars(&self) -> Vec<E::Fr> {
        self.scalars.clone()
    }
}

#[allow(unsafe_code)]
/// Wrapper over the MSM function to use the blstrs underlying function
pub fn msm_specific<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C::Curve]) -> C::Curve {
    // We empirically checked that for MSMs larger than 2**18, the blstrs
    // implementation regresses on x86.  On modern aarch64 (Apple M-series,
    // A17/A18) the blstrs fast path keeps winning well past 2^20 because its
    // hand-tuned Pippenger + threadpool dominate the generic `msm_best`.
    // Raise the gate to 2^21 so k=21 (n = 1 048 576 ≤ 2 097 152) stays on
    // the fast path.  See docs/k21-plan.md (proposal 1).
    if coeffs.len() <= (1 << 21) && TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>() {
        // Safe: we just checked type
        let coeffs = unsafe { &*(coeffs as *const _ as *const [Fq]) };
        let bases = unsafe { &*(bases as *const _ as *const [G1Projective]) };
        let res = G1Projective::multi_exp(bases, coeffs);
        unsafe { std::mem::transmute_copy(&res) }
    } else {
        // S3 (k21-iter-3): chunked-Pippenger fallback. The default
        // path allocates `affine_bases: Vec<C> = vec![identity; n]`
        // (192 MiB at n=2^21, 384 MiB at n=2^22) and then `msm_best`
        // internally allocates another `bases_local: Vec<Affine<F>>`
        // of the same size — at k=22 that's 768 MiB per MSM call.
        //
        // Pippenger composes naturally over chunks: split coeffs/bases
        // into fixed-size chunks, run msm_best on each, sum the
        // partial G1Projective results. Per-chunk peak heap drops to
        // ~24 MiB (256K bases × 96 B) regardless of total n. Trade:
        // a few % more CPU because msm_best's window size `c` is
        // chosen per-chunk and a smaller `c` does more total windows.
        //
        // Override the chunk size with `MIDNIGHT_MSM_CHUNK_LOG2` (an
        // integer; final chunk size = `1 << that`). Set to a large
        // value (e.g. 32) to disable chunking entirely.
        const DEFAULT_CHUNK_LOG2: u32 = 18;
        let chunk_log2: u32 = std::env::var("MIDNIGHT_MSM_CHUNK_LOG2")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_CHUNK_LOG2);
        let chunk: usize = 1usize << chunk_log2;

        if coeffs.len() <= chunk {
            let mut affine_bases = vec![C::identity(); coeffs.len()];
            C::Curve::batch_normalize(bases, &mut affine_bases);
            return msm_best(coeffs, &affine_bases);
        }

        let mut acc = C::Curve::identity();
        for (c_chunk, b_chunk) in coeffs.chunks(chunk).zip(bases.chunks(chunk)) {
            let mut affine_chunk = vec![C::identity(); c_chunk.len()];
            C::Curve::batch_normalize(b_chunk, &mut affine_chunk);
            acc = acc + msm_best(c_chunk, &affine_chunk);
            // affine_chunk + msm_best's bases_local drop here, freeing
            // ~48 MiB before the next iteration allocates.
        }
        acc
    }
}

/// Two channel MSM accumulator
#[derive(Debug, Clone)]
pub struct DualMSM<E: Engine> {
    pub(crate) left: MSMKZG<E>,
    pub(crate) right: MSMKZG<E>,
}

/// A [DualMSM] split into left and right vectors of `(Scalar, Point)` tuples
pub type SplitDualMSM<'a, E> = (
    Vec<(&'a <E as Engine>::Fr, &'a <E as Engine>::G1)>,
    Vec<(&'a <E as Engine>::Fr, &'a <E as Engine>::G1)>,
);

impl<E: MultiMillerLoop + Debug> Default for DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn default() -> Self {
        Self::init()
    }
}

impl<E: MultiMillerLoop> Guard<E::Fr, KZGCommitmentScheme<E>> for DualMSM<E>
where
    E::G1: Default + CurveExt<ScalarExt = E::Fr> + ProcessedSerdeObject,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn verify(
        self,
        params: &<KZGCommitmentScheme<E> as PolynomialCommitmentScheme<E::Fr>>::VerifierParameters,
    ) -> Result<(), Error> {
        self.check(params).then_some(()).ok_or(Error::OpeningError)
    }
}

impl<E: MultiMillerLoop + Debug> DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Create an empty two channel MSM accumulator instance
    pub fn init() -> Self {
        Self {
            left: MSMKZG::init(),
            right: MSMKZG::init(),
        }
    }

    /// Create a new two channel MSM accumulator instance
    pub fn new(left: MSMKZG<E>, right: MSMKZG<E>) -> Self {
        Self { left, right }
    }

    /// Split the [DualMSM] into `left` and `right`
    pub fn split(&self) -> SplitDualMSM<E> {
        let left = self.left.scalars.iter().zip(self.left.bases.iter()).collect();
        let right = self.right.scalars.iter().zip(self.right.bases.iter()).collect();
        (left, right)
    }

    /// Scale all scalars in the MSM by some scaling factor
    pub fn scale(&mut self, e: E::Fr) {
        self.left.scale(e);
        self.right.scale(e);
    }

    /// Add another multiexp into this one
    pub fn add_msm(&mut self, other: Self) {
        self.left.add_msm(&other.left);
        self.right.add_msm(&other.right);
    }

    /// Performs final pairing check with given verifier params and two channel
    /// linear combination
    pub fn check(self, params: &ParamsVerifierKZG<E>) -> bool {
        let left = if self.left.scalars.len() == 1 && self.left.scalars[0] == E::Fr::ONE {
            self.left.bases[0]
        } else {
            self.left.eval()
        };

        let right = self.right.eval();

        let (term_1, term_2) = (
            (&left.into(), &params.s_g2_prepared),
            (&right.into(), &params.n_g2_prepared),
        );
        let terms = &[term_1, term_2];

        bool::from(E::multi_miller_loop(&terms[..]).final_exponentiation().is_identity())
    }
}
