//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::FromUniformBytes;

use crate::{
    poly::{
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, PinnedEvaluationDomain,
        Polynomial,
    },
    transcript::{Hashable, Transcript},
    utils::{
        helpers::{
            byte_length, polynomial_slice_byte_length, read_polynomial_vec, write_polynomial_slice,
            ProcessedSerdeObject,
        },
        SerdeFormat,
    },
};

mod circuit;
mod error;
pub(crate) mod evaluation;
mod keygen;
pub(crate) mod lookup;
// S5 (P1, scaffold): mmap-backed PK loader infrastructure. Not yet
// wired into ProvingKey::read — see docs/k21-s5-mmap-pk-design.md.
pub(crate) mod mmap_pk;
pub mod permutation;
pub(crate) mod traces;
pub(crate) mod trash;
pub(crate) mod vanishing;

#[cfg(feature = "bench-internal")]
pub mod bench;

mod prover;
mod verifier;

use std::io;

pub use circuit::*;
pub use error::*;
pub(crate) use evaluation::Evaluator;
use ff::{PrimeField, WithSmallOrderMulGroup};
pub use keygen::*;
use midnight_curves::serde::SerdeObject;
pub use prover::*;
pub use verifier::*;

use crate::poly::commitment::PolynomialCommitmentScheme;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<F: PrimeField, CS: PolynomialCommitmentScheme<F>> {
    domain: EvaluationDomain<F>,
    fixed_commitments: Vec<CS::Commitment>,
    permutation: permutation::VerifyingKey<F, CS>,
    cs: ConstraintSystem<F>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: F,
}

// Current version of the VK
const VERSION: u8 = 0x03;

impl<F, CS> VerifyingKey<F, CS>
where
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    CS: PolynomialCommitmentScheme<F>,
{
    /// Returns `n`
    pub fn n(&self) -> u64 {
        self.domain.n
    }
    /// Writes a verifying key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in
    ///   standard form. Writes a field element in standard form, with
    ///   endianness specified by the `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in
    ///   Montgomery form Writes a field element into raw bytes in its internal
    ///   Montgomery representation, WITHOUT performing the expensive Montgomery
    ///   reduction.
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        // Version byte that will be checked on read.
        writer.write_all(&[VERSION])?;
        let k = &self.domain.k();
        assert!(*k <= F::S);
        // k value fits in 1 byte
        writer.write_all(&[*k as u8])?;
        writer.write_all(&(self.fixed_commitments.len() as u32).to_le_bytes())?;
        for commitment in &self.fixed_commitments {
            commitment.write(writer, format)?;
        }
        self.permutation.write(writer, format)?;

        Ok(())
    }

    /// Reads a verification key from a buffer for the associated [Circuit].
    ///
    /// Reads a curve element from the buffer and parses it according to the
    /// `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by
    ///   the `PrimeField` implementation, and checks that the element is less
    ///   than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in
    ///   Montgomery form. Checks that field elements are less than modulus, and
    ///   then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with
    ///   coordinates in Montgomery form; does not perform any checks.
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<F>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let mut cs = ConstraintSystem::default();
        #[cfg(feature = "circuit-params")]
        let _config = ConcreteCircuit::configure_with_params(&mut cs, params);
        #[cfg(not(feature = "circuit-params"))]
        let _config = ConcreteCircuit::configure(&mut cs);

        Self::read_from_cs(reader, format, cs)
    }

    /// Reads a verification key from a buffer, using the provided
    /// [ConstraintSystem].
    ///
    /// Reads a curve element from the buffer and parses it according to the
    /// `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by
    ///   the `PrimeField` implementation, and checks that the element is less
    ///   than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in
    ///   Montgomery form. Checks that field elements are less than modulus, and
    ///   then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with
    ///   coordinates in Montgomery form; does not perform any checks.
    pub fn read_from_cs<R: io::Read>(
        reader: &mut R,
        format: SerdeFormat,
        cs: ConstraintSystem<F>,
    ) -> io::Result<Self> {
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        if VERSION != version_byte[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected version byte",
            ));
        }

        let mut k = [0u8; 1];
        reader.read_exact(&mut k)?;
        let k = u8::from_le_bytes(k);
        if k as u32 > F::S {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("circuit size value (k): {} exceeds maxium: {}", k, F::S),
            ));
        }

        let domain = EvaluationDomain::new(cs.degree() as u32, k.into());

        let mut num_fixed_columns = [0u8; 4];
        reader.read_exact(&mut num_fixed_columns)?;
        let num_fixed_columns = u32::from_le_bytes(num_fixed_columns);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| CS::Commitment::read(reader, format))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation, format)?;

        // we still need to replace selectors with fixed Expressions in `cs`
        let fake_selectors = vec![vec![]; cs.num_selectors];
        let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors);

        Ok(Self::from_parts(domain, fixed_commitments, permutation, cs))
    }

    /// Writes a verifying key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length(format));
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a verification key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<F>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> VerifyingKey<F, CS> {
    /// Return the bytes_length of a VerifyingKey
    pub fn bytes_length(&self, format: SerdeFormat) -> usize {
        10 + (self.fixed_commitments.len() * byte_length::<CS::Commitment>(format))
            + self.permutation.bytes_length(format)
    }

    fn from_parts(
        domain: EvaluationDomain<F>,
        fixed_commitments: Vec<CS::Commitment>,
        permutation: permutation::VerifyingKey<F, CS>,
        cs: ConstraintSystem<F>,
    ) -> Self
    where
        F: FromUniformBytes<64>,
    {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: F::ZERO,
        };

        let mut hasher =
            Blake2bParams::new().hash_length(64).personal(b"Halo2-Verify-Key").to_state();

        // We serialise the commitments of the VK to get the `transcript_repr`.
        let mut buffer = Vec::new();
        buffer.push(VERSION);
        let k = &vk.domain.k();
        assert!(*k <= F::S);
        buffer.push(*k as u8);
        buffer.extend_from_slice(&(vk.fixed_commitments.len() as u32).to_le_bytes());
        for commitment in &vk.fixed_commitments {
            commitment
                .write(&mut buffer, SerdeFormat::RawBytesUnchecked)
                .expect("Failed to write to buffer - this is a bug.");
        }

        buffer.extend_from_slice(&(vk.permutation.commitments().len() as u32).to_le_bytes());
        for commitment in vk.permutation.commitments() {
            commitment
                .write(&mut buffer, SerdeFormat::RawBytesUnchecked)
                .expect("Failed to write to buffer - this is a bug.");
        }

        // We use the debug implementation to add the gates and domain to the hashed
        // buffer. We should eventually move away from debug implementation for
        // this purpose. See https://github.com/midnightntwrk/halo2/issues/5
        buffer.extend_from_slice(format!("{:?}", vk.get_domain().pinned()).as_bytes());
        buffer.extend_from_slice(format!("{:?}", vk.cs().pinned()).as_bytes());

        hasher.update(&buffer);

        // Hash in final Blake2bState
        vk.transcript_repr = F::from_uniform_bytes(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<T: Transcript>(&self, transcript: &mut T) -> io::Result<()>
    where
        F: Hashable<T::Hash>,
    {
        transcript.common(&self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, F, CS> {
        PinnedVerificationKey {
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Returns commitments of fixed polynomials
    pub fn fixed_commitments(&self) -> &Vec<CS::Commitment> {
        &self.fixed_commitments
    }

    /// Returns `VerifyingKey` of permutation
    pub fn permutation(&self) -> &permutation::VerifyingKey<F, CS> {
        &self.permutation
    }

    /// Returns `ConstraintSystem`
    pub fn cs(&self) -> &ConstraintSystem<F> {
        &self.cs
    }

    /// Returns representative of this `VerifyingKey` in transcripts
    pub fn transcript_repr(&self) -> F {
        self.transcript_repr
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, F: PrimeField, CS: PolynomialCommitmentScheme<F>> {
    domain: PinnedEvaluationDomain<'a, F>,
    cs: PinnedConstraintSystem<'a, F>,
    fixed_commitments: &'a Vec<CS::Commitment>,
    permutation: &'a permutation::VerifyingKey<F, CS>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<F: PrimeField, CS: PolynomialCommitmentScheme<F>> {
    pub(crate) vk: VerifyingKey<F, CS>,
    pub(crate) l0: Polynomial<F, ExtendedLagrangeCoeff>,
    pub(crate) l_last: Polynomial<F, ExtendedLagrangeCoeff>,
    pub(crate) l_active_row: Polynomial<F, ExtendedLagrangeCoeff>,
    pub(crate) fixed_values: Vec<Polynomial<F, LagrangeCoeff>>,
    pub(crate) fixed_polys: Vec<Polynomial<F, Coeff>>,
    pub(crate) fixed_cosets: Vec<Polynomial<F, ExtendedLagrangeCoeff>>,
    pub(crate) permutation: permutation::ProvingKey<F>,
    pub(crate) ev: Evaluator<F>,
    /// S5 (P2): optional mmap-backed view of `fixed_polys`. When
    /// `Some`, the corresponding `fixed_polys` Vec above is empty
    /// and callers MUST read polynomials through
    /// [`ProvingKey::fixed_polys_slice`]. Wrapped in `Arc` so PK
    /// `clone()` shares the underlying mmap + tempfile.
    pub(crate) fixed_polys_mmap: Option<std::sync::Arc<mmap_pk::MmappedPolys<F, Coeff>>>,

    /// S5 (P3): optional mmap-backed view of `fixed_values`
    /// (LagrangeCoeff basis, ~640 MiB at k=21). Same semantics as
    /// `fixed_polys_mmap`. Read via [`ProvingKey::fixed_values_slice`].
    pub(crate) fixed_values_mmap:
        Option<std::sync::Arc<mmap_pk::MmappedPolys<F, LagrangeCoeff>>>,

    /// S5 (P3): optional mmap-backed view of `permutation.polys`
    /// (Coeff basis, ~640 MiB at k=21). Same semantics as
    /// `fixed_polys_mmap`. Read via
    /// [`ProvingKey::permutation_polys_slice`]. Lives at the
    /// top-level PK rather than inside `permutation::ProvingKey`
    /// to keep the permutation type clone-safe and small.
    pub(crate) permutation_polys_mmap:
        Option<std::sync::Arc<mmap_pk::MmappedPolys<F, Coeff>>>,
}

// S5 (P2): bound-free impl so the accessor + spill are available
// from every code path that holds a `ProvingKey<F, CS>` — including
// the `prover::create_proof` path which doesn't constrain
// `F: FromUniformBytes<64>`.
impl<F: PrimeField, CS: PolynomialCommitmentScheme<F>> ProvingKey<F, CS> {
    /// S5 (P2): Read `fixed_polys` through a single accessor so the
    /// in-place mmap-backed sidecar can transparently replace the
    /// heap `Vec` when engaged.
    ///
    /// When `fixed_polys_mmap` is `Some`, the heap `fixed_polys`
    /// vector is empty and the polynomials live in mmap'd file
    /// pages. When `None`, the legacy heap vector is returned.
    /// Callers that previously indexed `&pk.fixed_polys[i]` should
    /// now use `&pk.fixed_polys_slice()[i]`.
    pub(crate) fn fixed_polys_slice(&self) -> &[Polynomial<F, Coeff>] {
        match self.fixed_polys_mmap.as_ref() {
            Some(m) => m.as_slice(),
            None => &self.fixed_polys,
        }
    }

    /// S5 (P3): mirror of `fixed_polys_slice` for `fixed_values`.
    pub(crate) fn fixed_values_slice(&self) -> &[Polynomial<F, LagrangeCoeff>] {
        match self.fixed_values_mmap.as_ref() {
            Some(m) => m.as_slice(),
            None => &self.fixed_values,
        }
    }

    /// S5 (P3): mirror of `fixed_polys_slice` for `permutation.polys`.
    pub(crate) fn permutation_polys_slice(&self) -> &[Polynomial<F, Coeff>] {
        match self.permutation_polys_mmap.as_ref() {
            Some(m) => m.as_slice(),
            None => &self.permutation.polys,
        }
    }

    /// S5 (P2): Move `fixed_polys` into mmap-backed storage.
    /// See [`Self::spill_all_to_mmap`] for a one-shot variant that
    /// also handles `fixed_values` and `permutation.polys` (P3).
    pub fn spill_fixed_polys_to_mmap(&mut self) -> std::io::Result<()> {
        if self.fixed_polys_mmap.is_some() {
            return Ok(());
        }
        if self.fixed_polys.is_empty() {
            return Ok(());
        }
        let n_per_poly = self.fixed_polys[0].values.len();
        let polys = std::mem::take(&mut self.fixed_polys);
        let mm = mmap_pk::spill_vec_to_disk(polys, n_per_poly)?;
        self.fixed_polys_mmap = Some(std::sync::Arc::new(mm));
        Ok(())
    }

    /// S5 (P3): Move `fixed_values` (LagrangeCoeff basis) into
    /// mmap-backed storage. Mirrors `spill_fixed_polys_to_mmap`.
    /// Target: ~640 MiB phys_footprint relief at k=21.
    pub fn spill_fixed_values_to_mmap(&mut self) -> std::io::Result<()> {
        if self.fixed_values_mmap.is_some() {
            return Ok(());
        }
        if self.fixed_values.is_empty() {
            return Ok(());
        }
        let n_per_poly = self.fixed_values[0].values.len();
        let polys = std::mem::take(&mut self.fixed_values);
        let mm = mmap_pk::spill_vec_to_disk(polys, n_per_poly)?;
        self.fixed_values_mmap = Some(std::sync::Arc::new(mm));
        Ok(())
    }

    /// S5 (P3): Move `permutation.polys` (Coeff basis) into
    /// mmap-backed storage. Sidecar lives at the top-level PK so
    /// `permutation::ProvingKey` stays clone-safe and small.
    /// Target: ~640 MiB phys_footprint relief at k=21.
    pub fn spill_permutation_polys_to_mmap(&mut self) -> std::io::Result<()> {
        if self.permutation_polys_mmap.is_some() {
            return Ok(());
        }
        if self.permutation.polys.is_empty() {
            return Ok(());
        }
        let n_per_poly = self.permutation.polys[0].values.len();
        let polys = std::mem::take(&mut self.permutation.polys);
        let mm = mmap_pk::spill_vec_to_disk(polys, n_per_poly)?;
        self.permutation_polys_mmap = Some(std::sync::Arc::new(mm));
        Ok(())
    }

    /// S5 (P3): one-shot — spill `fixed_polys`, `fixed_values`,
    /// AND `permutation.polys` in sequence. Each step is
    /// independently idempotent. Failures are best-effort: a
    /// failure at any step is returned but earlier steps stay
    /// applied (so partial relief is preserved).
    ///
    /// Combined target at k=21: ~2.6 GiB phys_footprint relief —
    /// pulls iOS below the iPhone 16 Pro jetsam threshold and
    /// makes real-device k=21 viable.
    pub fn spill_all_to_mmap(&mut self) -> std::io::Result<()> {
        self.spill_fixed_polys_to_mmap()?;
        self.spill_fixed_values_to_mmap()?;
        self.spill_permutation_polys_to_mmap()?;
        Ok(())
    }
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> ProvingKey<F, CS>
where
    F: FromUniformBytes<64>,
{
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<F, CS> {
        &self.vk
    }

    /// Gets the total number of bytes in the serialization of `self`
    pub fn bytes_length(&self, format: SerdeFormat) -> usize {
        self.vk.bytes_length(format)
            + 12 // bytes used for encoding the length(u32) of "l0", "l_last" & "l_active_row" polys
            + polynomial_slice_byte_length(&self.fixed_values)
            + self.permutation.bytes_length()
    }
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> ProvingKey<F, CS>
where
    F: PrimeField + FromUniformBytes<64> + SerdeObject,
{
    /// Writes a proving key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in
    ///   standard form. Writes a field element in standard form, with
    ///   endianness specified by the `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in
    ///   Montgomery form Writes a field element into raw bytes in its internal
    ///   Montgomery representation, WITHOUT performing the expensive Montgomery
    ///   reduction. Does so by first writing the verifying key and then
    ///   serializing the rest of the data (in the form of field polynomials)
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.vk.write(writer, format)?;
        write_polynomial_slice(&self.fixed_values, writer)?;
        self.permutation.write(writer)?;
        Ok(())
    }

    /// Reads a proving key from a buffer.
    /// Does so by reading verification key first, and then deserializing the
    /// rest of the file into the remaining proving key data.
    ///
    /// Reads a curve element from the buffer and parses it according to the
    /// `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by
    ///   the `PrimeField` implementation, and checks that the element is less
    ///   than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in
    ///   Montgomery form. Checks that field elements are less than modulus, and
    ///   then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with
    ///   coordinates in Montgomery form; does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<F>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let vk = VerifyingKey::<F, CS>::read::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        let [l0, l_last, l_active_row] = compute_lagrange_polys(&vk, &vk.cs);
        let fixed_values = read_polynomial_vec(reader, format)?;
        let fixed_polys: Vec<_> = fixed_values
            .iter()
            .map(|poly| vk.domain.lagrange_to_coeff(poly.clone()))
            .collect();
        let fixed_cosets = fixed_polys
            .iter()
            .map(|poly| vk.domain.coeff_to_extended(poly.clone()))
            .collect();
        let permutation =
            permutation::ProvingKey::read(reader, format, &vk.domain, &vk.cs.permutation)?;
        let ev = Evaluator::new(vk.cs());
        let mut pk = Self {
            vk,
            l0,
            l_last,
            l_active_row,
            fixed_values,
            fixed_polys,
            fixed_cosets,
            permutation,
            ev,
            fixed_polys_mmap: None,
            fixed_values_mmap: None,
            permutation_polys_mmap: None,
        };
        // S5 (P2+P3): opt-in mmap-spill of fixed_polys +
        // fixed_values + permutation.polys immediately after
        // deserialise. Gated on `MIDNIGHT_SPILL_PK=1` so the
        // default path stays bit-for-bit identical to the pre-S5
        // behaviour. iOS sim k=21: combined target ~2.6 GiB
        // of dirty-anon heap turned into clean file-backed pages
        // (no `phys_footprint` contribution under the jetsam
        // metric). Failures are best-effort — earlier successful
        // spills are kept; subsequent ones return the io::Error.
        if matches!(std::env::var("MIDNIGHT_SPILL_PK").as_deref(), Ok("1")) {
            let _ = pk.spill_all_to_mmap();
        }
        Ok(pk)
    }

    /// Writes a proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length(format));
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a proving key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<F>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<F: PrimeField, CS: PolynomialCommitmentScheme<F>> VerifyingKey<F, CS> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }
}
