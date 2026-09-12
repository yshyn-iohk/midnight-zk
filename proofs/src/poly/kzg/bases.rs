//! Storage abstraction for the SRS bases (`g`, `g_lagrange`).
//!
//! `BasesStorage<C>` wraps either an owned `Vec<C>` (the default,
//! used by `unsafe_setup` and the eager `read_custom` path) or a
//! slice view into a memory-mapped file. Both variants `Deref` to
//! `&[C]` so existing MSM call sites in `poly/kzg/mod.rs` need no
//! change.
//!
//! The mmap variant is used by `ParamsKZG::read_mmap_arc`. It
//! reduces the heap footprint of the SRS to ~zero — at k=20 (BLS
//! 12-381, ~144 B per `G1Projective`) we save ~144 MiB per
//! materialised basis. The file mapping itself still counts toward
//! RSS when its pages are *touched*, but the OS evicts cold pages
//! under memory pressure (during heavy FFT phases the SRS pages
//! aren't referenced and get reclaimed). That OS-paging dance is
//! the difference between a 5 GiB peak (process killed on Android)
//! and a 4 GiB peak (process survives) at k=20-22.
//!
//! ## Layout invariant
//!
//! `BasesStorage::Mapped` is sound only when constructed against a
//! file whose bytes are a verbatim contiguous sequence of `C`s in
//! the same memory layout the producer used. We control that
//! format via [`write_mmap_companion`](crate::poly::kzg::params::ParamsKZG::write_mmap_companion);
//! external callers must not feed arbitrary files to
//! `read_mmap_arc`.
//!
//! On BLS12-381 / midnight-curves the `G1Projective` /
//! `G1Affine` are both `#[repr(transparent)]` wrappers over a
//! C-bindings struct from `blst`, so the layout is platform-stable
//! and the cast is sound by construction.
//!
//! ## `unsafe` budget
//!
//! The crate-level `#![deny(unsafe_code)]` is opted out of for
//! this module — the mmap-as-slice cast and the manual
//! `Send`/`Sync` impls are the irreducible unsafe primitive that
//! the whole optimisation rests on. All `unsafe` blocks are
//! confined here behind invariants documented at construction.

#![allow(unsafe_code)]

use std::ops::Deref;
use std::sync::Arc;

use memmap2::Mmap;

/// Storage backing for an SRS basis vector.
/// View a slice of `C` as its raw bytes, for writing an mmap companion file.
///
/// The inverse of what [`BasesStorage::mapped`] reads back, and it lives here
/// for the same reason: this module is where the crate's `unsafe` budget is
/// spent, so the cast and the invariant that justifies it stay in one place.
///
/// Sound for any `C` — reading the bytes of a live `[C]` is always defined.
/// The *asymmetry* is that reading them back as `[C]` is only sound under the
/// layout invariant documented at the top of this module, which is why these
/// files are a local cache and not an interchange format.
/// Map `count` values of `C` out of `mmap` starting at byte `offset`.
///
/// The checked entry point to [`BasesStorage::Mapped`], and the only one
/// outside this module. It exists so callers need no `unsafe` of their own:
/// every precondition `mapped` states is verified here and reported as
/// `InvalidData` rather than assumed.
///
/// - the block lies wholly inside the mapping, with overflow-safe arithmetic
/// - the block start is aligned for `C`, which is **not** implied by the file
///   layout: block starts depend on `size_of::<C>()` and need not land on an
///   `align_of::<C>()` boundary for every `C`
/// - the `Arc` is cloned in, so the mapping outlives every borrow
///
/// What it cannot check is the layout invariant — that the bytes were written
/// from live `C` on a platform with this representation. That is the caller's,
/// and is why these files carry a magic and a point-size field.
pub(crate) fn map_block<C: 'static>(
    mmap: &Arc<Mmap>,
    offset: usize,
    count: usize,
) -> std::io::Result<BasesStorage<C>> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let size = std::mem::size_of::<C>();
    let end = count
        .checked_mul(size)
        .and_then(|n| offset.checked_add(n))
        .ok_or_else(|| bad("mmap block: size overflows"))?;
    if end > mmap.len() {
        return Err(bad("mmap block: extends past end of file"));
    }
    // SAFETY: `offset <= end <= mmap.len()`, so this stays within the mapping.
    let ptr = unsafe { mmap.as_ptr().add(offset) } as *const C;
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<C>()) {
        return Err(bad("mmap block: misaligned for this element type"));
    }
    // SAFETY: bounds and alignment checked immediately above; the Arc clone
    // keeps the mapping alive for as long as the storage, satisfying (4).
    Ok(unsafe { BasesStorage::mapped(Arc::clone(mmap), ptr, count) })
}

pub(crate) fn slice_as_bytes<C>(s: &[C]) -> &[u8] {
    // SAFETY: `s` is a live `[C]`, so `len * size_of::<C>()` bytes from its
    // start are initialised and owned by it, and `u8` has alignment 1. The
    // returned borrow carries `s`'s lifetime.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub(crate) enum BasesStorage<C: 'static> {
    /// Heap-allocated. The default for `unsafe_setup`, the eager
    /// `read_custom`, and any `g_to_lagrange` recompute.
    Owned(Vec<C>),
    /// Slice view into a memory-mapped file. The `Arc<Mmap>` keeps
    /// the mapping alive for the lifetime of this value.
    Mapped {
        /// Holds the mapping open. Cloning is cheap (`Arc::clone`)
        /// and increments the refcount; the mapping is dropped
        /// when the last clone falls out of scope.
        _mmap: Arc<Mmap>,
        /// Pointer into the mapping. Must be `align_of::<C>()`-
        /// aligned and remain valid for `len` consecutive `C`s.
        ptr: *const C,
        /// Number of `C`s reachable from `ptr`.
        len: usize,
    },
}

// SAFETY: the raw pointer in `Mapped` is sound to share across
// threads as long as `C: Send + Sync`. The mapping itself is
// Send + Sync (memmap2 doc); the pointer is immutable for the
// lifetime of the value.
unsafe impl<C: Send + Sync + 'static> Send for BasesStorage<C> {}
unsafe impl<C: Send + Sync + 'static> Sync for BasesStorage<C> {}

impl<C: 'static> BasesStorage<C> {
    /// Construct from an owned Vec — the common case.
    pub(crate) fn owned(v: Vec<C>) -> Self {
        BasesStorage::Owned(v)
    }

    /// Construct from a slice view into a memory-mapped file.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// 1. `ptr` points into the `mmap` region.
    /// 2. `ptr` has at least `align_of::<C>()` alignment.
    /// 3. The `len * size_of::<C>()` bytes starting at `ptr` are
    ///    a valid bit-pattern for `[C; len]` — i.e. the file was
    ///    produced by serialising live `C` values via the same
    ///    in-memory representation we're now claiming.
    /// 4. The `Arc<Mmap>` lives at least as long as any borrow
    ///    obtained through `Deref`.
    pub(crate) unsafe fn mapped(mmap: Arc<Mmap>, ptr: *const C, len: usize) -> Self {
        BasesStorage::Mapped {
            _mmap: mmap,
            ptr,
            len,
        }
    }

    /// `true` when the storage is backed by a memory-mapped file.
    /// Used by `downsize` to refuse in-place mutation.
    #[allow(dead_code)] // used once downsize is implemented for mapped
    pub(crate) fn is_mapped(&self) -> bool {
        matches!(self, BasesStorage::Mapped { .. })
    }

    /// Truncate to `n` elements, allocating-copying out of an mmap
    /// region if necessary. `Vec::truncate` is in-place; for the
    /// mmap variant we must materialise into an owned Vec because
    /// the file mapping is read-only.
    pub(crate) fn truncate_into_owned(&mut self, n: usize)
    where
        C: Clone,
    {
        match self {
            BasesStorage::Owned(v) => v.truncate(n),
            BasesStorage::Mapped { .. } => {
                let copy: Vec<C> = self.deref()[..n].to_vec();
                *self = BasesStorage::Owned(copy);
            }
        }
    }
}

impl<C: 'static> Deref for BasesStorage<C> {
    type Target = [C];
    fn deref(&self) -> &[C] {
        match self {
            BasesStorage::Owned(v) => v.as_slice(),
            // SAFETY: documented invariants in `mapped()` plus
            // `_mmap` keeps the region alive for our lifetime.
            BasesStorage::Mapped { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

impl<C: Clone + 'static> Clone for BasesStorage<C> {
    fn clone(&self) -> Self {
        match self {
            BasesStorage::Owned(v) => BasesStorage::Owned(v.clone()),
            // Cloning a Mapped is a refcount bump on the Arc plus
            // pointer copy — same backing region, no allocation.
            BasesStorage::Mapped { _mmap, ptr, len } => BasesStorage::Mapped {
                _mmap: Arc::clone(_mmap),
                ptr: *ptr,
                len: *len,
            },
        }
    }
}

impl<C: std::fmt::Debug + 'static> std::fmt::Debug for BasesStorage<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            BasesStorage::Owned(_) => "Owned",
            BasesStorage::Mapped { .. } => "Mapped",
        };
        f.debug_struct("BasesStorage")
            .field("kind", &kind)
            .field("len", &self.deref().len())
            .finish()
    }
}

impl<C: PartialEq + 'static> PartialEq for BasesStorage<C> {
    fn eq(&self, other: &Self) -> bool {
        // Compare via slice equality — backing storage doesn't
        // matter for value equality.
        self.deref() == other.deref()
    }
}

impl<C: Eq + 'static> Eq for BasesStorage<C> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_round_trips() {
        let bs: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3, 4]);
        assert_eq!(bs.len(), 4);
        assert_eq!(&*bs, &[1, 2, 3, 4]);
        assert!(!bs.is_mapped());
    }

    #[test]
    fn clone_preserves_data() {
        let bs: BasesStorage<u32> = BasesStorage::owned(vec![10, 20, 30]);
        let cloned = bs.clone();
        assert_eq!(&*bs, &*cloned);
    }

    #[test]
    fn truncate_into_owned_shortens() {
        let mut bs: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3, 4, 5]);
        bs.truncate_into_owned(3);
        assert_eq!(&*bs, &[1, 2, 3]);
    }

    #[test]
    fn equality_uses_slice() {
        let a: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3]);
        let b: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3]);
        let c: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
