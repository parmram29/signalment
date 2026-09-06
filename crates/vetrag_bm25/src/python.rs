//! PyO3 bindings. Plumbing only — no scoring logic belongs in this file.
//!
//! This layer is written for you; the work to fill in is in `bm25.rs`. It is
//! here so that the moment the core compiles, the Python side can import it.

use pyo3::prelude::*;

use crate::bm25::Bm25Index as CoreIndex;

/// A built BM25 index over chunk texts, searchable from Python.
///
/// Mirrors `vetrag.retrieval.bm25.BM25Retriever` exactly. "Exactly" is not a
/// figure of speech: `tests/test_backend_equivalence.py` asserts identical
/// rankings and near-identical scores against the Python implementation, and
/// that suite is the specification this crate is written against.
#[pyclass(module = "vetrag_bm25", name = "Bm25Index", frozen)]
pub struct PyBm25Index {
    inner: CoreIndex,
}

#[pymethods]
impl PyBm25Index {
    /// Build the index.
    ///
    /// Args mirror the Python constructor: parallel lists of chunk text,
    /// owning note id, and chunk id, plus the two BM25 parameters.
    #[new]
    #[pyo3(signature = (texts, note_ids, chunk_ids, k1=1.5, b=0.75))]
    fn new(
        texts: Vec<String>,
        note_ids: Vec<String>,
        chunk_ids: Vec<String>,
        k1: f64,
        b: f64,
    ) -> PyResult<Self> {
        if texts.len() != note_ids.len() || texts.len() != chunk_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "texts, note_ids and chunk_ids must be the same length",
            ));
        }
        Ok(Self {
            inner: CoreIndex::new(&texts, note_ids, chunk_ids, k1, b),
        })
    }

    /// Search, returning at most `k` results as `(note_id, score, chunk_id)`
    /// tuples, best first, already deduplicated by note.
    ///
    /// Returning plain tuples rather than a custom class keeps the boundary
    /// crossing cheap and leaves `Hit` construction to the Python side.
    #[pyo3(signature = (query, k=5))]
    fn search(&self, py: Python<'_>, query: &str, k: usize) -> Vec<(String, f64, String)> {
        // The GIL is not needed for scoring, and releasing it is most of the
        // point of doing this in Rust: other Python threads keep running while
        // a large corpus is scanned.
        py.detach(|| {
            self.inner
                .search(query, k)
                .into_iter()
                .map(|hit| (hit.note_id, hit.score, hit.chunk_id))
                .collect()
        })
    }

    /// Number of indexed chunks.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("<Bm25Index chunks={}>", self.inner.len())
    }
}

#[pymodule]
fn vetrag_bm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBm25Index>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
