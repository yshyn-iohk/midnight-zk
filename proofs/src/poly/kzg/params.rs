use std::{fmt::Debug, io};

use crate::poly::kzg::bases::{map_block, slice_as_bytes, BasesStorage};
use ff::{Field, PrimeField};
use group::{Curve, Group, GroupEncoding, prime::PrimeCurveAffine};
use midnight_curves::{
    pairing::{Engine, MultiMillerLoop},
    serde::SerdeObject,
};
use rand_core::RngCore;

use crate::{
    poly::{PolynomialBasis, PolynomialRepresentation, commitment::Params},
    utils::{
        SerdeFormat,
        arithmetic::{CurveAffine, g_to_lagrange, parallelize},
        helpers::ProcessedSerdeObject,
    },
};

/// These are the public parameters for the polynomial commitment scheme.
#[derive(Debug, Clone)]
pub struct ParamsKZG<E: Engine> {
    pub(crate) g: BasesStorage<E::G1Affine>,
    pub(crate) g_lagrange: BasesStorage<E::G1Affine>,
    /// Suffix-sum of `g_lagrange`:
    /// `g_lagrange_delta[i] = sum_{j=i}^{n-1} g_lagrange[j]`.
    pub(crate) g_lagrange_delta: BasesStorage<E::G1Affine>,
    /// Suffix-sum of `g_lagrange_delta`:
    /// `g_lagrange_double_delta[i] = sum_{j=i}^{n-1} g_lagrange_delta[j]`.
    /// Used as the SRS for the `LagrangeDoubleDelta` basis. Same derivation
    /// as `g_lagrange_delta`, applied a second time:
    /// `sum_j a_j · L_j = sum_i c_i · g_lagrange_double_delta[i]`,
    /// where `c_i = b_i - b_{i-1}` and `b_i = a_i - a_{i-1}`.
    pub(crate) g_lagrange_double_delta: BasesStorage<E::G1Affine>,
    pub(crate) g2: E::G2,
    pub(crate) s_g2: E::G2,
}

/// Builds the suffix-sum of an affine SRS: `out[i] = sum_{j=i}^{n-1} input[j]`.
///
/// Accumulate in projective coordinates (one curve add per step) and
/// batch-normalise to affine at the end. n-1 projective adds + one O(n)
/// batch_normalise.
///
/// # Panics
/// If input is empty
fn suffix_sum<C: CurveAffine>(input: &[C]) -> Vec<C> {
    let n = input.len();
    let mut acc_proj: Vec<C::Curve> = vec![C::Curve::identity(); n];
    acc_proj[n - 1] = input[n - 1].into();
    for i in (0..n - 1).rev() {
        acc_proj[i] = acc_proj[i + 1] + input[i];
    }
    let mut acc_affine = vec![C::identity(); n];
    C::Curve::batch_normalize(&acc_proj, &mut acc_affine);
    acc_affine
}

impl<E: Engine> Params for ParamsKZG<E>
where
    E::G1Affine: CurveAffine,
{
    fn max_k(&self) -> u32 {
        #[cfg(not(feature = "single-h-commitment"))]
        assert_eq!(self.g.len(), self.g_lagrange.len());
        self.g_lagrange.len().ilog2()
    }

    fn g_monomial_size(&self) -> usize {
        self.g.len()
    }

    fn downsize(&mut self, new_k: u32) {
        ParamsKZG::<E>::downsize(self, new_k)
    }
}

impl<E: Engine + Debug> ParamsKZG<E>
where
    E::G1Affine: CurveAffine,
{
    /// Return the SRS bases corresponding to the polynomial representation `B`.
    pub fn bases<B: PolynomialRepresentation>(&self) -> &[E::G1Affine] {
        match B::BASIS {
            PolynomialBasis::Coeff => &self.g,
            PolynomialBasis::Lagrange => &self.g_lagrange,
            PolynomialBasis::LagrangeDelta => &self.g_lagrange_delta,
            PolynomialBasis::LagrangeDoubleDelta => &self.g_lagrange_double_delta,
            PolynomialBasis::ExtendedLagrange => {
                unimplemented!("KZG does not support extended Lagrange bases")
            }
        }
    }


    /// Write the SRS in the layout [`read_mmap_arc`](Self::read_mmap_arc) expects.
    ///
    /// Format **v2**. v1 carried two bases; upstream's `ParamsKZG` now holds four
    /// — `g`, `g_lagrange`, and the two suffix-sum bases — so the layout gained
    /// two more blocks and the magic changed to refuse v1 files rather than
    /// silently misread them.
    ///
    /// ```text
    /// 0    "MDNGHTV2"  magic
    /// 8    u32         version = 2
    /// 12   u32         k
    /// 16   u32         size_of::<E::G1Affine>()
    /// 20   u32         flags (bit 0: all four bases present)
    /// 24   [u8; 8]     reserved
    /// 32   4 x (u64 offset, u64 count)   g, g_lagrange, delta, double_delta
    /// 96   u64 g2_offset, u64 g2_size, u64 s_g2_offset, u64 s_g2_size
    /// 256  basis blocks, then g2, then s_g2
    /// ```
    ///
    /// Blocks start at 256 for SIMD-friendly alignment, which also leaves room
    /// to grow the header again without moving the data.
    ///
    /// The basis blocks are the raw in-memory bytes of consecutive
    /// `E::G1Affine`. That is only sound to read back on a platform with the
    /// same representation — see the layout invariant on
    /// [`BasesStorage`](crate::poly::kzg::bases::BasesStorage). These files are
    /// a local cache, not an interchange format.
    pub fn write_mmap_companion<W: io::Write>(&self, writer: &mut W) -> io::Result<()>
    where
        E::G2: ProcessedSerdeObject,
    {
        const HEADER_LEN: usize = 128;
        const BLOCK_OFFSET: u64 = 256;

        let point_size = std::mem::size_of::<E::G1Affine>() as u64;
        let k = (self.g.len() as u64).ilog2();

        let bases: [&[E::G1Affine]; 4] = [
            &self.g,
            &self.g_lagrange,
            &self.g_lagrange_delta,
            &self.g_lagrange_double_delta,
        ];

        // Lay the four blocks out back to back from BLOCK_OFFSET.
        let mut offsets = [0u64; 4];
        let mut counts = [0u64; 4];
        let mut cursor = BLOCK_OFFSET;
        for (i, b) in bases.iter().enumerate() {
            offsets[i] = cursor;
            counts[i] = b.len() as u64;
            cursor += b.len() as u64 * point_size;
        }

        // g2 / s_g2 are small and serialised, not mapped.
        let mut g2_buf = Vec::new();
        self.g2.write(&mut g2_buf, SerdeFormat::RawBytesUnchecked)?;
        let mut s_g2_buf = Vec::new();
        self.s_g2.write(&mut s_g2_buf, SerdeFormat::RawBytesUnchecked)?;
        let g2_offset = cursor;
        let s_g2_offset = g2_offset + g2_buf.len() as u64;

        writer.write_all(b"MDNGHTV2")?;
        writer.write_all(&2u32.to_le_bytes())?;
        writer.write_all(&k.to_le_bytes())?;
        writer.write_all(&(point_size as u32).to_le_bytes())?;
        writer.write_all(&1u32.to_le_bytes())?;
        writer.write_all(&[0u8; 8])?;
        for i in 0..4 {
            writer.write_all(&offsets[i].to_le_bytes())?;
            writer.write_all(&counts[i].to_le_bytes())?;
        }
        writer.write_all(&g2_offset.to_le_bytes())?;
        writer.write_all(&(g2_buf.len() as u64).to_le_bytes())?;
        writer.write_all(&s_g2_offset.to_le_bytes())?;
        writer.write_all(&(s_g2_buf.len() as u64).to_le_bytes())?;
        writer.write_all(&vec![0u8; BLOCK_OFFSET as usize - HEADER_LEN])?;

        for b in bases.iter() {
            writer.write_all(slice_as_bytes(b))?;
        }
        writer.write_all(&g2_buf)?;
        writer.write_all(&s_g2_buf)?;
        Ok(())
    }


    /// Construct `ParamsKZG` whose four bases are slice views over a
    /// memory-mapped companion file written by
    /// [`write_mmap_companion`](Self::write_mmap_companion).
    ///
    /// The point of the exercise: at k=20 the four bases are the bulk of the
    /// SRS heap, and mapping them moves that cost from resident memory to page
    /// cache the OS can reclaim under pressure. On a phone that is the
    /// difference between a peak the kernel tolerates and one it does not.
    ///
    /// Returns `InvalidData` rather than panicking on a file that is not ours,
    /// is a different format version, was written on a platform with a
    /// different point representation, is truncated, or whose blocks are not
    /// correctly aligned for `E::G1Affine`. Those are all reachable by handing
    /// it an arbitrary file, so none of them may be an `unwrap`.
    pub fn read_mmap_arc(mmap: std::sync::Arc<memmap2::Mmap>) -> io::Result<Self>
    where
        E::G2: ProcessedSerdeObject,
    {
        const HEADER_LEN: usize = 128;
        let bytes: &[u8] = &mmap[..];
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());

        if bytes.len() < HEADER_LEN {
            return Err(bad("mmap companion: shorter than its header"));
        }
        if &bytes[0..8] != b"MDNGHTV2" {
            return Err(bad("mmap companion: bad magic (v1 files are not readable)"));
        }
        let rd32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let rd64 = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());

        if rd32(8) != 2 {
            return Err(bad("mmap companion: unsupported format version"));
        }
        let point_size = rd32(16) as usize;
        if point_size != std::mem::size_of::<E::G1Affine>() {
            return Err(bad("mmap companion: point size differs from this build"));
        }

        let mut bases: Vec<BasesStorage<E::G1Affine>> = Vec::with_capacity(4);
        for i in 0..4 {
            let off = rd64(32 + i * 16) as usize;
            let count = rd64(32 + i * 16 + 8) as usize;
            bases.push(map_block::<E::G1Affine>(&mmap, off, count)?);
        }

        let g2_off = rd64(96) as usize;
        let g2_size = rd64(104) as usize;
        let s_g2_off = rd64(112) as usize;
        let s_g2_size = rd64(120) as usize;
        if g2_off + g2_size > bytes.len() || s_g2_off + s_g2_size > bytes.len() {
            return Err(bad("mmap companion: g2 block extends past end of file"));
        }
        let g2 = E::G2::read(&mut &bytes[g2_off..g2_off + g2_size], SerdeFormat::RawBytesUnchecked)?;
        let s_g2 =
            E::G2::read(&mut &bytes[s_g2_off..s_g2_off + s_g2_size], SerdeFormat::RawBytesUnchecked)?;

        let mut it = bases.into_iter();
        Ok(ParamsKZG {
            g: it.next().unwrap(),
            g_lagrange: it.next().unwrap(),
            g_lagrange_delta: it.next().unwrap(),
            g_lagrange_double_delta: it.next().unwrap(),
            g2,
            s_g2,
        })
    }

    /// Downsize the current parameters to match a smaller `k`.
    pub fn downsize(&mut self, new_k: u32) {
        if self.max_k() == new_k {
            return;
        }

        let n = 1 << new_k;
        assert!(n < self.g_lagrange.len());
        self.g.truncate_into_owned(n);
        self.g_lagrange = BasesStorage::owned(g_to_lagrange(&self.g, new_k));
        self.g_lagrange_delta = BasesStorage::owned(suffix_sum(&self.g_lagrange));
        self.g_lagrange_double_delta = BasesStorage::owned(suffix_sum(&self.g_lagrange_delta));
    }

    /// Recompute the Lagrange basis for a smaller circuit domain `new_k` while
    /// keeping the full monomial basis `g` intact.
    ///
    /// Use this when the `single-h-commitment` feature is enabled: generate an
    /// SRS large enough for the whole quotient polynomial (i.e. with `k'`
    /// such that `2^{k'} ≥ (n-1) * quotient_poly_degree`), then call
    /// `downsize_lagrange(k)` so that `max_k()` equals the circuit domain size
    /// `k` while `g` retains its original length for the H-polynomial
    /// commitment.
    pub fn downsize_lagrange(&mut self, new_k: u32) {
        let n = 1usize << new_k;
        assert!(
            self.g.len() >= n,
            "g is too small to build a Lagrange basis of size 2^{new_k}"
        );
        self.g_lagrange = BasesStorage::owned(g_to_lagrange(&self.g[..n], new_k));
        self.g_lagrange_delta = BasesStorage::owned(suffix_sum(&self.g_lagrange));
        self.g_lagrange_double_delta = BasesStorage::owned(suffix_sum(&self.g_lagrange_delta));
    }

    /// Combine the monomial basis from `extended` with the Lagrange basis from
    /// `self`, consuming both. This avoids the FFT that `downsize_lagrange`
    /// would otherwise require.
    ///
    /// # Panics
    ///
    /// If `extended.g` is not strictly larger than `self.g`, or if the shared
    /// prefix of the monomial bases does not match.
    pub fn with_extended_monomial(mut self, extended: Self) -> Self {
        assert!(
            extended.g.len() > self.g.len(),
            "extended SRS must be strictly larger than the base SRS"
        );
        assert!(
            self.g[..] == extended.g[..self.g.len()],
            "monomial bases of the two SRSs do not match"
        );
        self.g = extended.g;
        self
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

        let mut g_affine = vec![E::G1Affine::identity(); n as usize];
        E::G1::batch_normalize(&g, &mut g_affine);

        let mut g_lagrange_affine = vec![E::G1Affine::identity(); n as usize];
        E::G1::batch_normalize(&g_lagrange, &mut g_lagrange_affine);

        let g_lagrange_delta = suffix_sum(&g_lagrange_affine);
        let g_lagrange_double_delta = suffix_sum(&g_lagrange_delta);

        let g2 = E::G2::generator();
        let s_g2 = g2 * s;

        Self {
            g: BasesStorage::owned(g_affine),
            g_lagrange: BasesStorage::owned(g_lagrange_affine),
            g_lagrange_delta: BasesStorage::owned(g_lagrange_delta),
            g_lagrange_double_delta: BasesStorage::owned(g_lagrange_double_delta),
            g2,
            s_g2,
        }
    }

    /// Initializes parameters for the curve through existing parameters
    /// k, g, g_lagrange (optional), g2, s_g2
    pub fn from_parts(
        k: u32,
        g: Vec<E::G1>,
        g_lagrange: Option<Vec<E::G1>>,
        g2: E::G2,
        s_g2: E::G2,
    ) -> Self {
        let n = g.len();
        let mut g_affine = vec![E::G1Affine::identity(); n];
        E::G1::batch_normalize(&g, &mut g_affine);
        let g_lagrange_affine = match g_lagrange {
            Some(g_l) => {
                let mut aff = vec![E::G1Affine::identity(); g_l.len()];
                E::G1::batch_normalize(&g_l, &mut aff);
                aff
            }
            None => g_to_lagrange(&g_affine, k),
        };
        let g_lagrange_delta = suffix_sum(&g_lagrange_affine);
        let g_lagrange_double_delta = suffix_sum(&g_lagrange_delta);
        Self {
            g: BasesStorage::owned(g_affine),
            g_lagrange: BasesStorage::owned(g_lagrange_affine),
            g_lagrange_delta: BasesStorage::owned(g_lagrange_delta),
            g_lagrange_double_delta: BasesStorage::owned(g_lagrange_double_delta),
            g2,
            s_g2,
        }
    }

    /// Returns the committed lagrange polynomials of these KZG params.
    pub fn g_lagrange(&self) -> &[E::G1Affine] {
        &self.g_lagrange
    }

    /// Returns generator on G2
    pub fn g2(&self) -> E::G2 {
        self.g2
    }

    /// Returns \[τ\]₂, a commitment to τ in G2.
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer
    pub fn write_custom<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        E::G1Affine: SerdeObject,
        E::G2: ProcessedSerdeObject,
    {
        writer.write_all(&self.g.len().ilog2().to_le_bytes())?;
        for el in self.g.iter() {
            match format {
                SerdeFormat::Processed => writer.write_all(el.to_bytes().as_ref())?,
                _ => el.write_raw(writer)?,
            }
        }
        for el in self.g_lagrange.iter() {
            match format {
                SerdeFormat::Processed => writer.write_all(el.to_bytes().as_ref())?,
                _ => el.write_raw(writer)?,
            }
        }
        self.g2.write(writer, format)?;
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read_custom<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self>
    where
        E::G1Affine: SerdeObject,
        E::G2: ProcessedSerdeObject,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        let (g, g_lagrange) = match format {
            SerdeFormat::Processed => {
                let load_points_from_file_parallelly =
                    |reader: &mut R| -> io::Result<Vec<E::G1Affine>> {
                        let mut points_compressed =
                            vec![<E::G1Affine as GroupEncoding>::Repr::default(); n];
                        for points_compressed in points_compressed.iter_mut() {
                            reader.read_exact((*points_compressed).as_mut())?;
                        }
                        let mut points = vec![Option::<E::G1Affine>::None; n];
                        parallelize(&mut points, |points, chunks| {
                            for (i, point) in points.iter_mut().enumerate() {
                                *point = Option::from(E::G1Affine::from_bytes(
                                    &points_compressed[chunks + i],
                                ));
                            }
                        });
                        points
                            .into_iter()
                            .map(|p| p.ok_or_else(|| io::Error::other("invalid point encoding")))
                            .collect()
                    };

                let g = load_points_from_file_parallelly(reader)?;
                let g_lagrange = load_points_from_file_parallelly(reader)?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytes => {
                let g =
                    (0..n).map(|_| E::G1Affine::read_raw(reader)).collect::<Result<Vec<_>, _>>()?;
                let g_lagrange =
                    (0..n).map(|_| E::G1Affine::read_raw(reader)).collect::<Result<Vec<_>, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytesUnchecked => {
                let g = (0..n).map(|_| E::G1Affine::read_raw_unchecked(reader)).collect::<Vec<_>>();
                let g_lagrange =
                    (0..n).map(|_| E::G1Affine::read_raw_unchecked(reader)).collect::<Vec<_>>();
                (g, g_lagrange)
            }
        };

        let g2 = E::G2::read(reader, format)?;
        let s_g2 = E::G2::read(reader, format)?;

        let g_lagrange_delta = suffix_sum(&g_lagrange);
        let g_lagrange_double_delta = suffix_sum(&g_lagrange_delta);
        Ok(Self {
            g: BasesStorage::owned(g),
            g_lagrange: BasesStorage::owned(g_lagrange),
            g_lagrange_delta: BasesStorage::owned(g_lagrange_delta),
            g_lagrange_double_delta: BasesStorage::owned(g_lagrange_double_delta),
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
    /// Returns \[τ\]₂, a commitment to τ in G2.
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

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
            PolynomialLabel,
            commitment::PolynomialCommitmentScheme,
            kzg::{KZGCommitmentScheme, params::ParamsKZG},
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

        let tmp = KZGCommitmentScheme::commit(&params, &a, PolynomialLabel::NoLabel);
        let com = KZGCommitmentScheme::commit(&params, &b, PolynomialLabel::NoLabel);

        assert_eq!(tmp, com);
    }

    #[test]
    fn test_commit_in_delta_basis_matches_lagrange() {
        const K: u32 = 6;

        use ff::Field;
        use midnight_curves::{Bls12, Fq};

        use crate::poly::EvaluationDomain;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let domain = EvaluationDomain::new(1, K);

        // A polynomial with long constant runs and a few isolated transitions —
        // exercises the structure the optimization is designed for.
        let mut a = domain.empty_lagrange();
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = match i {
                0..16 => Fq::ZERO,
                16..40 => Fq::from(7),
                40..50 => Fq::from(7) + Fq::from(i as u64),
                _ => Fq::from(42),
            };
        }

        let c_lagrange = KZGCommitmentScheme::commit(&params, &a, PolynomialLabel::NoLabel);
        let c_delta = KZGCommitmentScheme::commit(&params, &a.to_delta(), PolynomialLabel::NoLabel);
        assert_eq!(c_lagrange, c_delta);

        // Round-trip identity: to_delta then into_lagrange recovers the original.
        let restored = a.to_delta().into_lagrange();
        assert_eq!(restored.values, a.values);
    }

    #[test]
    fn test_commit_in_double_delta_basis_matches_lagrange() {
        const K: u32 = 6;

        use midnight_curves::{Bls12, Fq};

        use crate::poly::EvaluationDomain;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let domain = EvaluationDomain::new(1, K);

        // A polynomial with linear runs (constant first differences) and a
        // few transitions — exercises the structure the double-delta basis
        // is designed for: linear segments collapse to zeros after two
        // delta applications.
        let mut a = domain.empty_lagrange();
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = match i {
                0..16 => Fq::from(i as u64),
                16..40 => Fq::from(16) + Fq::from(2u64) * Fq::from((i - 16) as u64),
                40..50 => Fq::from(99),
                _ => Fq::from(i as u64) - Fq::from(50),
            };
        }

        let c_lagrange = KZGCommitmentScheme::commit(&params, &a, PolynomialLabel::NoLabel);

        // Fused single-pass conversion matches the two-step path.
        let c_double_delta_fused =
            KZGCommitmentScheme::commit(&params, &a.to_double_delta(), PolynomialLabel::NoLabel);
        let c_double_delta_two_step = KZGCommitmentScheme::commit(
            &params,
            &a.to_delta().into_double_delta(),
            PolynomialLabel::NoLabel,
        );
        assert_eq!(c_lagrange, c_double_delta_fused);
        assert_eq!(c_lagrange, c_double_delta_two_step);

        // Both fused and two-step transforms produce the same scalar vector.
        let fused = a.to_double_delta();
        let two_step = a.to_delta().into_double_delta();
        assert_eq!(fused.values, two_step.values);

        // Round-trip identity, both inverse paths.
        let restored_fused = a.to_double_delta().into_lagrange();
        let restored_stepwise = a.to_double_delta().into_lagrange_delta().into_lagrange();
        assert_eq!(restored_fused.values, a.values);
        assert_eq!(restored_stepwise.values, a.values);
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
        assert_eq!(params0.g_lagrange.len(), params1.g_lagrange.len());

        assert_eq!(params0.g, params1.g);
        assert_eq!(params0.g_lagrange, params1.g_lagrange);
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);
    }

    #[test]
    // `Mmap::map` is unsafe by design: the mapping is UB if the file is mutated
    // externally while live. Wrapping it in a "safe" helper would launder that,
    // so it is allowed here instead, where the file is test-local and untouched.
    #[allow(unsafe_code)]
    fn mmap_companion_round_trips_all_four_bases() {
        use midnight_curves::Bls12;
        let params = ParamsKZG::<Bls12>::unsafe_setup(4, OsRng);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("srs.mmap");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            params.write_mmap_companion(&mut f).unwrap();
        }

        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: test-local file, not mutated while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let mapped = ParamsKZG::<Bls12>::read_mmap_arc(std::sync::Arc::new(mmap)).unwrap();

        // All four bases must survive, not just the two the original patch carried.
        assert_eq!(&*mapped.g, &*params.g, "g");
        assert_eq!(&*mapped.g_lagrange, &*params.g_lagrange, "g_lagrange");
        assert_eq!(&*mapped.g_lagrange_delta, &*params.g_lagrange_delta, "delta");
        assert_eq!(
            &*mapped.g_lagrange_double_delta, &*params.g_lagrange_double_delta,
            "double_delta"
        );
        assert_eq!(mapped.g2, params.g2);
        assert_eq!(mapped.s_g2, params.s_g2);
    }

    #[test]
    // `Mmap::map` is unsafe by design: the mapping is UB if the file is mutated
    // externally while live. Wrapping it in a "safe" helper would launder that,
    // so it is allowed here instead, where the file is test-local and untouched.
    #[allow(unsafe_code)]
    fn mmap_companion_rejects_bad_input_instead_of_panicking() {
        use midnight_curves::Bls12;
        let params = ParamsKZG::<Bls12>::unsafe_setup(4, OsRng);
        let mut good = Vec::new();
        params.write_mmap_companion(&mut good).unwrap();

        let load = |bytes: Vec<u8>| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("f");
            std::fs::write(&path, &bytes).unwrap();
            let f = std::fs::File::open(&path).unwrap();
            let m = unsafe { memmap2::Mmap::map(&f) }.unwrap();
            ParamsKZG::<Bls12>::read_mmap_arc(std::sync::Arc::new(m))
        };

        // Each of these is reachable by handing the loader an arbitrary file,
        // so each must be an error rather than a panic or a wrong read.
        assert!(load(vec![0u8; 16]).is_err(), "too short for a header");

        let mut wrong_magic = good.clone();
        wrong_magic[0..8].copy_from_slice(b"MDNGHTV1");
        assert!(load(wrong_magic).is_err(), "v1 magic must be refused, not misread");

        let mut wrong_version = good.clone();
        wrong_version[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(load(wrong_version).is_err(), "unknown version");

        let mut wrong_point = good.clone();
        wrong_point[16..20].copy_from_slice(&7u32.to_le_bytes());
        assert!(load(wrong_point).is_err(), "point size from another platform");

        let truncated = good[..good.len() / 2].to_vec();
        assert!(load(truncated).is_err(), "block past end of file");

        let mut huge_count = good.clone();
        huge_count[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(load(huge_count).is_err(), "count that overflows on multiply");
    }
}
