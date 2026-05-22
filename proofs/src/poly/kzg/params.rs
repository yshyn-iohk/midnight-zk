use std::{fmt::Debug, io, sync::OnceLock};

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use midnight_curves::pairing::{Engine, MultiMillerLoop};
use rand_core::RngCore;

use crate::{
    poly::commitment::Params,
    utils::{
        arithmetic::{g_to_lagrange, parallelize},
        helpers::ProcessedSerdeObject,
        SerdeFormat,
    },
};

/// These are the public parameters for the polynomial commitment scheme.
///
/// `g_lagrange` is the inverse-FFT of `g`. It is a *derived* value, not
/// independent SRS material — anything we know about `g` determines
/// `g_lagrange` exactly. To keep the resident SRS as small as possible on
/// memory-constrained targets (mobile, wasm), `g_lagrange` is wrapped in
/// `OnceLock`: callers can release it via [`drop_lazy_bases`](Self::drop_lazy_bases)
/// between proofs and the next `commit_lagrange` (or other Lagrange-basis
/// access via [`g_lagrange_slice`](Self::g_lagrange_slice)) will rebuild it
/// from `g`.
///
/// File-format compatibility is preserved: `read_custom` still consumes
/// the `g_lagrange` bytes the writer produced and pre-populates the lock,
/// so callers see no behaviour change unless they opt in to dropping. We
/// document the byte layout so a future `read_custom_lazy` can `Seek` past
/// the Lagrange block to avoid even paying the load-time cost.
#[derive(Debug, Clone)]
pub struct ParamsKZG<E: Engine> {
    pub(crate) g: Vec<E::G1>,
    pub(crate) g_lagrange: OnceLock<Vec<E::G1>>,
    pub(crate) g2: E::G2,
    pub(crate) s_g2: E::G2,
}

impl<E: Engine> Params for ParamsKZG<E>
where
    E::G1: Curve,
{
    fn max_k(&self) -> u32 {
        // `g_lagrange` may not yet be initialised — use `g` as the source
        // of truth for the SRS size. (When initialised, `g_lagrange.len()`
        // always equals `g.len()`.)
        self.g.len().ilog2()
    }

    fn downsize(&mut self, new_k: u32) {
        ParamsKZG::<E>::downsize(self, new_k)
    }
}

impl<E: Engine + Debug> ParamsKZG<E>
where
    E::G1: Curve,
{
    /// Returns the Lagrange-basis SRS as a slice. The first call after
    /// construction or after [`drop_lazy_bases`](Self::drop_lazy_bases)
    /// runs `g_to_lagrange(&g, log2(g.len()))` and caches the result.
    ///
    /// Internal hot-path accessor — `commit_lagrange` calls this rather
    /// than touching the field directly so the lazy-init contract is
    /// honoured uniformly.
    pub fn g_lagrange_slice(&self) -> &[E::G1] {
        self.g_lagrange
            .get_or_init(|| {
                let k = self.g.len().ilog2();
                g_to_lagrange(&self.g, k)
            })
            .as_slice()
    }

    /// Release the cached Lagrange-basis SRS so the prover footprint
    /// drops back to just the monomial basis `g`. The next call to
    /// `commit_lagrange` (or any other `g_lagrange_slice` consumer) will
    /// recompute via FFT — pay CPU to win RAM.
    ///
    /// At BLS12-381 / projective storage the per-element cost is 144 B,
    /// so at k=20 this releases roughly 144 MiB. Recompute cost is one
    /// inverse-NTT over a 2^k domain (≈ tens-of-ms range on M-class
    /// silicon at k=14, scaling near-linearly).
    pub fn drop_lazy_bases(&mut self) {
        self.g_lagrange = OnceLock::new();
    }

    /// Downsize the current parameters to match a smaller `k`.
    pub fn downsize(&mut self, new_k: u32) {
        if self.max_k() == new_k {
            return;
        }

        let n = 1 << new_k;
        assert!((n as u64) < (1u64 << self.max_k()));
        self.g.truncate(n);
        // Cached Lagrange basis is now stale (it was sized for the old
        // `g`). Reset; next consumer recomputes against the truncated
        // monomial basis.
        self.g_lagrange = OnceLock::new();
    }

    /// Initializes parameters for the curve, draws toxic secret from given rng.
    /// MUST NOT be used in production.
    pub fn unsafe_setup<R: RngCore>(k: u32, rng: R) -> Self {
        // Largest root of unity exponent of the Engine is `2^E::Fr::S`, so we can
        // only support FFTs of polynomials below degree `2^E::Fr::S`.
        assert!(k <= E::Fr::S);
        let n: u64 = 1 << k;

        // Calculate g = [G1, [s] G1, [s^2] G1, ..., [s^(n-1)] G1] in parallel.
        let g1 = E::G1::generator();
        let s = <E::Fr>::random(rng);

        let mut g = vec![E::G1::identity(); n as usize];
        parallelize(&mut g, |g, start| {
            let mut current_g: E::G1 = g1;
            current_g *= s.pow_vartime([start as u64]);
            for g in g.iter_mut() {
                *g = current_g;
                current_g *= s;
            }
        });

        let mut g_lagrange = vec![E::G1::identity(); n as usize];
        let mut root = E::Fr::ROOT_OF_UNITY;
        for _ in k..E::Fr::S {
            root = root.square();
        }
        let n_inv = E::Fr::from(n).invert().expect("inversion should be ok for n = 1<<k");
        let multiplier = (s.pow_vartime([n]) - E::Fr::ONE) * n_inv;
        parallelize(&mut g_lagrange, |g, start| {
            for (idx, g) in g.iter_mut().enumerate() {
                let offset = start + idx;
                let root_pow = root.pow_vartime([offset as u64]);
                let scalar = multiplier * root_pow * (s - root_pow).invert().unwrap();
                *g = g1 * scalar;
            }
        });

        let g2 = E::G2::generator();
        let s_g2 = g2 * s;

        Self {
            g,
            // Eagerly cache the Lagrange basis here — `unsafe_setup`
            // computed it anyway (line above), and skipping the cache
            // would just force recompute on first use with no win.
            g_lagrange: OnceLock::from(g_lagrange),
            g2,
            s_g2,
        }
    }

    /// Initializes parameters for the curve through existing parameters
    /// k, g, g_lagrange (optional), g2, s_g2.
    ///
    /// When `g_lagrange` is `None` the Lagrange basis is deferred — it
    /// will be computed on first access. Pass `Some` only if you already
    /// have the values and want to skip the load-time FFT.
    pub fn from_parts(
        k: u32,
        g: Vec<E::G1>,
        g_lagrange: Option<Vec<E::G1>>,
        g2: E::G2,
        s_g2: E::G2,
    ) -> Self {
        let _ = k; // historically used to drive the FFT; now derived from g.len()
        Self {
            g,
            g_lagrange: match g_lagrange {
                Some(g_l) => OnceLock::from(g_l),
                None => OnceLock::new(),
            },
            g2,
            s_g2,
        }
    }

    /// Returns the committed lagrange polynomials of these KZG params.
    ///
    /// Lazily materialises the basis if it has not been computed yet
    /// (or was released via [`drop_lazy_bases`](Self::drop_lazy_bases)).
    pub fn g_lagrange(&self) -> &[E::G1] {
        self.g_lagrange_slice()
    }

    /// Returns generator on G2
    pub fn g2(&self) -> E::G2 {
        self.g2
    }

    /// Returns first power of secret on G2
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer.
    ///
    /// Forces the Lagrange basis to be materialised before writing so the
    /// on-disk layout `[k, g[..], g_lagrange[..], g2, s_g2]` stays
    /// byte-for-byte compatible with v0.7.0 readers. Callers who want to
    /// elide the Lagrange block from disk should use a separate writer
    /// (none provided yet).
    pub fn write_custom<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        writer.write_all(&self.g.len().ilog2().to_le_bytes())?;
        for el in self.g.iter() {
            el.write(writer, format)?;
        }
        for el in self.g_lagrange_slice().iter() {
            el.write(writer, format)?;
        }
        self.g2.write(writer, format)?;
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read_custom<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        let (g, g_lagrange) = match format {
            SerdeFormat::Processed => {
                use group::GroupEncoding;
                let load_points_from_file_parallelly =
                    |reader: &mut R| -> io::Result<Vec<Option<E::G1>>> {
                        let mut points_compressed =
                            vec![<<E as Engine>::G1 as GroupEncoding>::Repr::default(); n];
                        for points_compressed in points_compressed.iter_mut() {
                            reader.read_exact((*points_compressed).as_mut())?;
                        }

                        let mut points = vec![Option::<E::G1>::None; n];
                        parallelize(&mut points, |points, chunks| {
                            for (i, point) in points.iter_mut().enumerate() {
                                *point =
                                    Option::from(E::G1::from_bytes(&points_compressed[chunks + i]));
                            }
                        });
                        Ok(points)
                    };

                let g = load_points_from_file_parallelly(reader)?;
                let g: Vec<<E as Engine>::G1> = g
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                let g_lagrange = load_points_from_file_parallelly(reader)?;
                let g_lagrange: Vec<<E as Engine>::G1> = g_lagrange
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytes => {
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytesUnchecked => {
                // avoid try branching for performance
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                (g, g_lagrange)
            }
        };

        let g2 = E::G2::read(reader, format)?;
        let s_g2 = E::G2::read(reader, format)?;

        // We just paid the I/O + parse cost to materialise `g_lagrange`
        // from disk — initialise the lock with it so the first commit
        // doesn't redundantly recompute. Callers who want to release
        // the basis between proofs invoke `drop_lazy_bases`.
        Ok(Self {
            g,
            g_lagrange: OnceLock::from(g_lagrange),
            g2,
            s_g2,
        })
    }
}

// TODO: see the issue at https://github.com/appliedzkp/halo2/issues/45
// So we probably need much smaller verifier key. However for new bases in g1
// should be in verifier keys.
/// KZG multi-open verification parameters
#[derive(Clone, Debug)]
pub struct ParamsVerifierKZG<E: MultiMillerLoop> {
    pub(crate) s_g2: E::G2,
    pub(crate) n_g2_prepared: E::G2Prepared,
    pub(crate) s_g2_prepared: E::G2Prepared,
}

impl<E: MultiMillerLoop + Debug> ParamsVerifierKZG<E>
where
    E::G2: Curve + ProcessedSerdeObject,
{
    /// Writes parameters to buffer
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let s_g2 = E::G2::read(reader, format)?;
        let s_g2_prepared = E::G2Prepared::from(s_g2.into());
        let n_g2_prepared = E::G2Prepared::from(-E::G2Affine::generator());

        Ok(Self {
            s_g2,
            n_g2_prepared,
            s_g2_prepared,
        })
    }
}

impl<E: MultiMillerLoop + Debug> ParamsKZG<E> {
    /// Consume the prover parameters into verifier parameters. Need to specify
    /// the size of public inputs.
    pub fn verifier_params(&self) -> ParamsVerifierKZG<E> {
        let n_g2_prepared = E::G2Prepared::from((-self.g2).into());
        let s_g2_prepared = E::G2Prepared::from(self.s_g2.into());
        ParamsVerifierKZG {
            s_g2: self.s_g2,
            n_g2_prepared,
            s_g2_prepared,
        }
    }
}

#[cfg(test)]
mod test {
    use rand_core::OsRng;

    use crate::{
        poly::{
            commitment::PolynomialCommitmentScheme,
            kzg::{params::ParamsKZG, KZGCommitmentScheme},
        },
        utils::SerdeFormat,
    };

    #[test]
    fn test_commit_lagrange() {
        const K: u32 = 6;

        use midnight_curves::{Bls12, Fq};

        use crate::poly::EvaluationDomain;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let domain = EvaluationDomain::new(1, K);

        let mut a = domain.empty_lagrange();

        for (i, a) in a.iter_mut().enumerate() {
            *a = Fq::from(i as u64);
        }

        let b = domain.lagrange_to_coeff(a.clone());

        let tmp = KZGCommitmentScheme::commit_lagrange(&params, &a);
        let commitment = KZGCommitmentScheme::commit(&params, &b);

        assert_eq!(commitment, tmp);
    }

    #[test]
    fn test_parameter_serialisation_roundtrip() {
        const K: u32 = 4;

        use midnight_curves::Bls12;

        let params0: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let mut data = vec![];
        ParamsKZG::write_custom(&params0, &mut data, SerdeFormat::RawBytesUnchecked).unwrap();
        let params1 =
            ParamsKZG::<Bls12>::read_custom::<_>(&mut &data[..], SerdeFormat::RawBytesUnchecked)
                .unwrap();

        assert_eq!(params0.g.len(), params1.g.len());
        // Materialise the Lagrange basis on both sides before comparing —
        // the field is `OnceLock` now, so we go through the lazy accessor.
        assert_eq!(params0.g_lagrange_slice().len(), params1.g_lagrange_slice().len());

        assert_eq!(params0.g, params1.g);
        assert_eq!(params0.g_lagrange_slice(), params1.g_lagrange_slice());
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);
    }

    /// Verifies that the lazy Lagrange basis matches an eager
    /// computation, and that `drop_lazy_bases` + `g_lagrange_slice`
    /// round-trips back to the same values via FFT.
    #[test]
    fn test_drop_lazy_bases_round_trip() {
        const K: u32 = 5;

        use midnight_curves::Bls12;

        use crate::poly::EvaluationDomain;

        let mut params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let _ = EvaluationDomain::<midnight_curves::Fq>::new(1, K);

        // Force-materialise via the accessor.
        let eager: Vec<_> = params.g_lagrange_slice().to_vec();
        assert_eq!(eager.len(), 1usize << K);

        // Drop the cache, then recompute from `g` via FFT — must equal.
        params.drop_lazy_bases();
        let recomputed: &[midnight_curves::G1Projective] = params.g_lagrange_slice();
        assert_eq!(eager.as_slice(), recomputed);
    }
}
