//! S5 — generic mmap-backed polynomial spill (foundation for the
//! mmap-backed Proving Key loader).
//!
//! ## What this module provides
//!
//! [`MmappedPolys<F, B>`] — a holder for a batch of `Polynomial<F, B>`
//! values that live in a tempfile mmap'd into the process address
//! space. The Polynomial views into the mmap are wrapped in
//! `ManuallyDrop` so the global allocator never tries to `free` a
//! pointer it doesn't own; the mapping (and the underlying tempfile)
//! are cleaned up when the holder is dropped.
//!
//! [`spill_iter_to_disk`] — the canonical spill primitive. Consumes
//! an iterator of `Polynomial<F, B>`, streams each one's raw bytes to
//! a tempfile (one polynomial at a time so peak heap stays at ~1
//! poly), mmaps the file, and returns an `MmappedPolys` view.
//!
//! [`spill_with_transform`] — convenience that applies a per-element
//! transform (e.g. `coeff_to_extended`) on the fly so the SpilledCosets
//! pattern from `plonk::prover` can be expressed as a one-liner over
//! this primitive.
//!
//! ## Why this is here
//!
//! The same trick that `prover::SpilledCosets` uses to back live
//! `evaluate_h` cosets with file pages is precisely what we need at
//! PK LOAD time to back the heaviest `ProvingKey` fields — see
//! `docs/k21-s5-mmap-pk-design.md`. Lifting the pattern to a generic
//! over polynomial basis `B` lets a single helper serve every
//! caller (live cosets during prove + persistent PK polys at load).
//!
//! Note on `B`: the basis marker is a `PhantomData` parameter — it
//! contributes no bytes to the on-disk layout. The same tempfile
//! could in principle be reinterpreted under a different basis, but
//! the public API keeps the typing strong so the compiler enforces
//! the invariant.

use std::{io, marker::PhantomData, mem::ManuallyDrop, sync::Arc};

use crate::poly::Polynomial;

/// Read-only batch of polynomials whose `values: Vec<F>` storage is
/// a non-owning view into a mmap'd tempfile.
///
/// The struct holds three pieces, in this drop order:
///   1. `polys` — `Vec<ManuallyDrop<Polynomial<F, B>>>`. Dropping the
///      vec frees the outer Vec spine but does NOT run the inner
///      Polynomial destructors (those would call `free` on mmap
///      pointers).
///   2. `_mmap` — `Arc<Mmap>`. When the refcount hits zero the OS
///      unmaps the file pages.
///   3. `_tmp_path` — `tempfile::TempPath`. Drop deletes the
///      underlying file from disk.
///
/// `Sync`/`Send`: the inner `Polynomial<F, B>` values are read-only
/// views; the struct is `Send + Sync` whenever `F: Send + Sync` and
/// `B: Send + Sync`. (The compiler derives these automatically from
/// the field types.)
pub struct MmappedPolys<F, B> {
    polys: Vec<ManuallyDrop<Polynomial<F, B>>>,
    _mmap: Arc<memmap2::Mmap>,
    _tmp_path: tempfile::TempPath,
}

impl<F, B> std::fmt::Debug for MmappedPolys<F, B> {
    // Custom Debug so callers that hold `MmappedPolys` as a field of
    // a `derive(Debug)` struct compile. We deliberately do NOT print
    // the polynomial values — they live in mmap pages, printing them
    // would force the OS to read the entire file into RAM.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmappedPolys")
            .field("n_polys", &self.polys.len())
            .field("mmap_bytes", &self._mmap.len())
            .finish()
    }
}

impl<F, B> MmappedPolys<F, B> {
    /// Borrow the polynomials as a flat slice. The returned slice is
    /// `&[Polynomial<F, B>]` (not `&[ManuallyDrop<...>]`) — sound
    /// because `ManuallyDrop<T>` is `#[repr(transparent)]` over `T`
    /// and callers only get shared, read-only access (no moves or
    /// drops through the slice).
    pub fn as_slice(&self) -> &[Polynomial<F, B>] {
        // SAFETY: ManuallyDrop<T> is #[repr(transparent)]; layout is
        // identical to T. Shared borrow, no drops, no mutations.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(
                self.polys.as_ptr() as *const Polynomial<F, B>,
                self.polys.len(),
            )
        }
    }

    /// Number of polynomials held.
    // P3 consumers (ProvingKey integration) call this; suppress the
    // dead-code warning while only the P1 surface ships.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.polys.len()
    }

    /// Whether the holder is empty.
    // P3 consumers (ProvingKey integration) call this; suppress the
    // dead-code warning while only the P1 surface ships.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }
}

/// Resolve the tempfile directory used by spills.
///
/// Honours `MIDNIGHT_SPILL_DIR` (useful when the default `TMPDIR`
/// partition is too small for the working set — e.g. Android
/// emulator `/data/local/tmp`). Falls back to the OS default when
/// the env var is unset or empty.
fn make_tempfile() -> io::Result<tempfile::NamedTempFile> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("midnight-spill-");
    match std::env::var("MIDNIGHT_SPILL_DIR") {
        Ok(dir) if !dir.is_empty() => builder.tempfile_in(dir),
        _ => builder.tempfile(),
    }
}

/// Stream each element of `polys` to a tempfile (raw field bytes,
/// no encoding), mmap the file read-only, and return Polynomial
/// views over the mapped pages.
///
/// All polynomials produced by the iterator MUST have
/// `values.len() == n_per_poly`. The invariant lets the read-back
/// compute the i-th polynomial's offset as `i * n_per_poly` without
/// per-element headers; it's `debug_assert!`-checked.
///
/// Peak transient heap stays at ~1 polynomial during the write — the
/// iterator yields owned `Polynomial<F, B>` values and each is
/// dropped immediately after its bytes hit the writer.
///
/// # Safety contract on `F`
///
/// `F` must be `Copy` and have a stable, plain-old-data
/// byte-for-byte in-memory representation (no interior pointers,
/// no padding holes that observably matter). This is satisfied by
/// `midnight-curves` field elements, which are `#[repr(transparent)]`
/// wrappers over `[u64; 4]`.
///
/// # Errors
///
/// Returns the underlying `io::Error` from tempfile creation, the
/// write loop, the flush, or `Mmap::map`.
pub fn spill_iter_to_disk<F, B, I>(polys: I, n_per_poly: usize) -> io::Result<MmappedPolys<F, B>>
where
    I: IntoIterator<Item = Polynomial<F, B>>,
{
    use std::io::Write as _;

    let elem_size = std::mem::size_of::<F>();
    let mut tmp = make_tempfile()?;

    // Streaming write: drop each `p` before allocating the next.
    {
        let mut writer = std::io::BufWriter::new(&mut tmp);
        for p in polys {
            debug_assert_eq!(
                p.values.len(),
                n_per_poly,
                "spill_iter_to_disk: polynomial size mismatch (got {}, expected n_per_poly = {})",
                p.values.len(),
                n_per_poly
            );
            // SAFETY: F is repr-stable plain-old-data per the
            // module-level safety contract. We borrow the raw bytes
            // of a live Vec<F> we own; the borrow ends before `p`
            // drops at the end of this loop iteration.
            #[allow(unsafe_code)]
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    p.values.as_ptr() as *const u8,
                    p.values.len() * elem_size,
                )
            };
            writer.write_all(bytes)?;
            // `p` drops here, freeing the polynomial's heap storage
            // before the next iteration's transient allocation.
        }
        writer.flush()?;
    }

    let (file, tmp_path) = tmp.into_parts();
    // SAFETY: read-only mmap of a file we just wrote and exclusively
    // own. No other process can be modifying it concurrently.
    #[allow(unsafe_code)]
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mmap_arc = Arc::new(mmap);

    // Build non-owning Polynomial views into the mmap region. The
    // number of polynomials is recovered from the file size (the
    // iterator is already exhausted at this point).
    let total_bytes = mmap_arc.len();
    debug_assert!(
        n_per_poly == 0 || total_bytes % (n_per_poly * elem_size) == 0,
        "spill_iter_to_disk: tempfile size {} not a multiple of {} × {} bytes",
        total_bytes,
        n_per_poly,
        elem_size
    );
    let n_polys = if n_per_poly == 0 {
        0
    } else {
        total_bytes / (n_per_poly * elem_size)
    };

    let mut polys_view = Vec::with_capacity(n_polys);
    let base_ptr = mmap_arc.as_ptr() as *const F;
    for i in 0..n_polys {
        // SAFETY: the file holds exactly `n_polys` consecutive
        // blocks of `n_per_poly × elem_size` bytes (verified by the
        // debug_assert above). The pointer arithmetic stays inside
        // the mmap region. The Vec is immediately wrapped in
        // ManuallyDrop so its allocator-aware destructor never runs.
        #[allow(unsafe_code)]
        let ptr = unsafe { base_ptr.add(i * n_per_poly) as *mut F };
        #[allow(unsafe_code)]
        let values = unsafe { Vec::from_raw_parts(ptr, n_per_poly, n_per_poly) };
        let poly = Polynomial::<F, B> {
            values,
            _marker: PhantomData,
        };
        polys_view.push(ManuallyDrop::new(poly));
    }

    Ok(MmappedPolys {
        polys: polys_view,
        _mmap: mmap_arc,
        _tmp_path: tmp_path,
    })
}

/// Convenience: spill a `Vec<Polynomial<F, B>>` (consuming it).
///
/// Equivalent to `spill_iter_to_disk(polys.into_iter(), n_per_poly)`
/// but the `Vec` is `drain(..)`'d in place so the Vec spine itself
/// can be freed mid-loop on long batches.
// P3 consumers (ProvingKey integration) call this; suppress the
// dead-code warning while only the P1 surface ships.
#[allow(dead_code)]
pub fn spill_vec_to_disk<F, B>(
    mut polys: Vec<Polynomial<F, B>>,
    n_per_poly: usize,
) -> io::Result<MmappedPolys<F, B>> {
    spill_iter_to_disk(polys.drain(..), n_per_poly)
}

/// Spill the result of applying `transform` to each input polynomial.
///
/// Useful when the on-disk form differs from the input form — for
/// instance the "cosets spill" case from `plonk::prover`, where the
/// input is `Polynomial<F, Coeff>` (small) but the stored form is
/// `Polynomial<F, ExtendedLagrangeCoeff>` (4× larger, after
/// `coeff_to_extended`).
///
/// Peak transient heap stays at ~1 OUTPUT polynomial — the transform
/// runs eagerly per element, the bytes are written, then the
/// transient polynomial drops before the next iteration.
///
/// Inputs are taken by shared reference so the caller retains the
/// originals (e.g. `pk.fixed_polys`, `pk.permutation.polys`) — the
/// transform clones internally as needed.
pub fn spill_with_transform<F, In, Out, T>(
    inputs: &[Polynomial<F, In>],
    n_per_out_poly: usize,
    mut transform: T,
) -> io::Result<MmappedPolys<F, Out>>
where
    T: FnMut(&Polynomial<F, In>) -> Polynomial<F, Out>,
{
    spill_iter_to_disk(inputs.iter().map(|p| transform(p)), n_per_out_poly)
}

// Integration tests against real `Polynomial<Fr, _>` values land in
// P3 once the type is wired into `ProvingKey::read`. The vendor
// crate's `dev-deps` pipeline is currently unbuildable in this fork,
// so unit tests with synthetic field types can't run here either —
// the consumer-side tests (smoke-test `proof_bytes` equality at
// k=10..18 with/without S5) are the actual coverage.
