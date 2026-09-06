//! Accelerated BM25 for the signalment retrieval harness.
//!
//! The crate is split deliberately:
//!
//! * [`bm25`] is pure Rust with no PyO3 in it at all. That is where the
//!   scoring lives, and it is testable with `cargo test --no-default-features`
//!   without libpython, Python headers, or an interpreter present.
//! * [`python`] is the PyO3 binding layer and contains no algorithm.
//!
//! Keeping the boundary sharp is what makes the core independently testable,
//! and it means a bug is always in one place or the other, never smeared
//! across both.

mod bm25;

pub use bm25::{tokenize, Bm25Index, Hit};

#[cfg(feature = "python")]
mod python;
