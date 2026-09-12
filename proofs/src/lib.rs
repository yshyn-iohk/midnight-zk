//! # midnight_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

// A host-only feature on a target that cannot honour it is a build error, not
// a runtime surprise. Without this the crate compiles for wasm with `mmap`
// enabled and then fails when the path is first taken - in a browser, at the
// worst possible moment, with a message that says nothing about features.
#[cfg(all(feature = "mmap", target_family = "wasm"))]
compile_error!(
    "feature `mmap` requires mmap(2), which wasm does not provide. \
     Build without it: the SRS is read eagerly instead."
);

#[cfg(all(feature = "disk-spill", target_family = "wasm"))]
compile_error!(
    "feature `disk-spill` requires a writable filesystem, which wasm does not \
     provide. Build without it: cosets are held on the heap instead."
);
pub mod circuit;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
pub mod utils;
