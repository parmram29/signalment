//! Okapi BM25 scoring. Pure Rust — no PyO3 in this file.
//!
//! # What you are implementing
//!
//! A port of `src/signalment/retrieval/bm25.py`. The Python version stays in the
//! repo permanently as the reference implementation; this is an accelerator,
//! not a replacement. `tests/test_backend_equivalence.py` asserts that the two
//! produce **identical rankings** and scores agreeing to ~1e-9, so the Python
//! file is the specification. When in doubt, read it — not this comment.
//!
//! # Scoring
//!
//! For a query term `t` in document `d`:
//!
//! ```text
//!                                    f(t,d) * (k1 + 1)
//!   score(t,d) = idf(t) * ─────────────────────────────────────────
//!                          f(t,d) + k1 * (1 - b + b * |d| / avgdl)
//!
//!   idf(t) = max( ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)), 0 )
//! ```
//!
//! `f(t,d)` is raw term frequency, `|d|` document length in tokens, `avgdl`
//! the mean document length, `N` the document count, `df(t)` the number of
//! documents containing `t`.
//!
//! The `k1` term controls frequency saturation (how fast repeated occurrences
//! stop adding score) and `b` controls length normalisation (how much a long
//! document is penalised). Worth understanding rather than copying — they are
//! the two knobs that matter, and Phase 3 of the blueprint asks you to know
//! what they do.
//!
//! # Four places this is easy to get subtly wrong
//!
//! Each of these is a real divergence the differential test will catch, and
//! each is worth reading before you start:
//!
//! 1. **Repeated query terms count more than once.** The Python loops over the
//!    token *list*, not a set, so `"cat cat"` scores a matching document twice
//!    as high as `"cat"`. Do not deduplicate query terms.
//! 2. **`idf` is floored at zero.** A term appearing in every document yields a
//!    negative raw idf; clamping it to 0 means such a term contributes nothing
//!    rather than *subtracting* score from documents that contain it.
//! 3. **An empty index must not divide by zero.** Python uses `avg_len or 1`,
//!    substituting 1.0 when `avg_len` is 0.0.
//! 4. **Deduplication keeps the maximum.** Several chunks share a note id; the
//!    note's score is the best of its chunks, not the sum and not the first.

use std::collections::HashMap;

/// One search result, already collapsed to its parent note.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub note_id: String,
    pub score: f64,
    pub chunk_id: String,
}

/// A built BM25 index over chunk texts.
///
/// All statistics are computed once in [`Bm25Index::new`]; `search` only reads.
pub struct Bm25Index {
    /// Parent note id per document, parallel to `tf`.
    note_ids: Vec<String>,
    /// Chunk id per document, parallel to `tf`.
    chunk_ids: Vec<String>,
    /// Term frequencies per document.
    tf: Vec<HashMap<String, u32>>,
    /// Token count per document.
    lengths: Vec<usize>,
    /// Mean of `lengths`, or 0.0 for an empty index.
    avg_len: f64,
    /// Inverse document frequency per term, floored at 0.0.
    idf: HashMap<String, f64>,
    k1: f64,
    b: f64,
}

/// Split text into lowercase word tokens.
///
/// Must match the Python regex `[a-z0-9][a-z0-9\-']*` applied to `text.lower()`
/// exactly. In practice that means:
///
/// * lowercase the input first (Unicode-aware, like Python's `str.lower()`);
/// * a token **starts** with an ASCII letter or digit;
/// * a token **continues** with an ASCII letter, digit, `-`, or `'`;
/// * every other character, including non-ASCII letters, is a separator.
///
/// Dosages and hyphenated drug names are the reason `-` and digits are kept:
/// `"0.5-1 mg/kg"` must produce `["0", "5-1", "mg", "kg"]`. Those are exactly
/// the tokens a dense retriever handles worst, so the lexical side must not
/// destroy them.
pub fn tokenize(text: &str) -> Vec<String> {
    // TODO(you): implement.
    //
    // 1. Lowercase the input.
    // 2. Walk the characters. When you see an ASCII alphanumeric and are not
    //    already inside a token, start one.
    // 3. While inside a token, accept ASCII alphanumeric, '-' and '\''.
    // 4. Any other character ends the current token.
    // 5. Do not forget the token still open when the input ends.
    //
    // Watch the trailing separator case: "5-1 " and "5-1" must tokenize the
    // same. The unit tests below cover it.
    let _ = text;
    todo!("tokenize")
}

impl Bm25Index {
    /// Build the index from parallel slices of chunk text, note id and chunk id.
    ///
    /// # Panics
    /// Callers must pass equal-length inputs; the PyO3 layer validates this and
    /// raises `ValueError` before reaching here.
    pub fn new(
        texts: &[String],
        note_ids: Vec<String>,
        chunk_ids: Vec<String>,
        k1: f64,
        b: f64,
    ) -> Self {
        debug_assert_eq!(texts.len(), note_ids.len());
        debug_assert_eq!(texts.len(), chunk_ids.len());

        // TODO(you): implement.
        //
        // 1. Tokenize every text once. Do not keep the token vectors around
        //    afterwards — only `tf` and `lengths` are needed at search time,
        //    and holding the raw tokens roughly doubles the index's memory.
        // 2. Build `tf`: a HashMap<String, u32> of term -> count per document.
        // 3. Build `lengths`: total token count per document (with duplicates).
        // 4. `avg_len` = mean of `lengths`, or 0.0 when there are no documents.
        // 5. Build document frequency: for each document, count each *distinct*
        //    term once. A HashMap<String, u32> accumulated over
        //    `tf[i].keys()` is the straightforward way.
        // 6. Convert document frequency to `idf` with the formula in the module
        //    docs, clamping negatives to 0.0 (see pitfall 2).
        //
        // `f64::ln` is the natural log. `n` is the document count as f64.
        let _ = (texts, &note_ids, &chunk_ids, k1, b);
        todo!("Bm25Index::new")
    }

    /// Score `query` against every document, returning at most `k` hits,
    /// best first, deduplicated by note id.
    ///
    /// Documents scoring zero are omitted entirely — a non-match is not a
    /// result with a low score, it is not a result.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        // TODO(you): implement.
        //
        // 1. Tokenize the query. Keep duplicates (see pitfall 1).
        // 2. For each document i, accumulate the per-term score from the module
        //    docs. Skip terms absent from that document's `tf`. Use
        //    `self.idf.get(term).copied().unwrap_or(0.0)` for a term that was
        //    never indexed.
        // 3. Substitute 1.0 for `avg_len` when it is 0.0 (see pitfall 3).
        // 4. Collect only documents with score > 0.0.
        // 5. Deduplicate by `note_id`, keeping the highest-scoring chunk and
        //    that chunk's id (see pitfall 4).
        // 6. Sort by score descending, breaking ties by `note_id` ascending so
        //    the output is deterministic across runs. `f64` is not `Ord`, so
        //    reach for `sort_by` with `partial_cmp`, not `sort`.
        // 7. Truncate to `k`.
        let _ = (query, k);
        todo!("Bm25Index::search")
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.tf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small index from `(text, note_id)` pairs, chunk ids generated.
    fn index(docs: &[(&str, &str)]) -> Bm25Index {
        let texts: Vec<String> = docs.iter().map(|(t, _)| t.to_string()).collect();
        let note_ids: Vec<String> = docs.iter().map(|(_, n)| n.to_string()).collect();
        let chunk_ids: Vec<String> = (0..docs.len()).map(|i| format!("c{i}")).collect();
        Bm25Index::new(&texts, note_ids, chunk_ids, 1.5, 0.75)
    }

    mod tokenizer {
        use super::*;

        #[test]
        fn lowercases() {
            assert_eq!(tokenize("Permethrin TOXICOSIS"), ["permethrin", "toxicosis"]);
        }

        #[test]
        fn keeps_digits_and_internal_hyphens() {
            // The exact case from the Python test suite.
            assert_eq!(tokenize("0.5-1 mg/kg"), ["0", "5-1", "mg", "kg"]);
        }

        #[test]
        fn keeps_apostrophes() {
            assert_eq!(tokenize("cushing's disease"), ["cushing's", "disease"]);
        }

        #[test]
        fn token_may_not_start_with_hyphen() {
            // The regex requires an alphanumeric first character, so a leading
            // hyphen is a separator, not part of the token.
            assert_eq!(tokenize("-abc"), ["abc"]);
        }

        #[test]
        fn closes_token_at_end_of_input() {
            assert_eq!(tokenize("abc"), ["abc"]);
            assert_eq!(tokenize("abc "), ["abc"]);
        }

        #[test]
        fn non_ascii_letters_separate_tokens() {
            // `[a-z0-9]` is ASCII-only even after lowercasing, so accented
            // characters split a word rather than joining it.
            assert_eq!(tokenize("ab\u{e9}cd"), ["ab", "cd"]);
        }

        #[test]
        fn empty_input() {
            assert!(tokenize("").is_empty());
            assert!(tokenize("   ...   ").is_empty());
        }
    }

    mod scoring {
        use super::*;

        #[test]
        fn finds_the_matching_document() {
            let idx = index(&[("permethrin sodium channels", "n1"), ("insulin glucose", "n2")]);
            let hits = idx.search("permethrin", 5);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].note_id, "n1");
            assert!(hits[0].score > 0.0);
        }

        #[test]
        fn non_matching_query_returns_nothing() {
            let idx = index(&[("permethrin", "n1")]);
            assert!(idx.search("cruciate ligament", 5).is_empty());
        }

        #[test]
        fn term_in_every_document_contributes_nothing() {
            // idf floored at zero: "cat" is in both, so it cannot separate them
            // and must not drag either score negative.
            let idx = index(&[("cat insulin", "n1"), ("cat permethrin", "n2")]);
            assert!(idx.search("cat", 5).is_empty());
        }

        #[test]
        fn repeated_query_terms_count_twice() {
            let idx = index(&[("permethrin toxicosis", "n1"), ("insulin", "n2")]);
            let once = idx.search("permethrin", 5)[0].score;
            let twice = idx.search("permethrin permethrin", 5)[0].score;
            assert!((twice - 2.0 * once).abs() < 1e-9, "{twice} != 2 * {once}");
        }

        #[test]
        fn results_sorted_by_score_descending() {
            let idx = index(&[
                ("permethrin permethrin permethrin", "n1"),
                ("permethrin", "n2"),
                ("unrelated", "n3"),
            ]);
            let hits = idx.search("permethrin", 5);
            assert_eq!(hits.len(), 2);
            assert!(hits[0].score >= hits[1].score);
        }

        #[test]
        fn respects_k() {
            let idx = index(&[("a x", "n1"), ("a y", "n2"), ("a z", "n3"), ("q", "n4")]);
            assert!(idx.search("x y z", 2).len() <= 2);
        }

        #[test]
        fn deduplicates_by_note_keeping_the_best_chunk() {
            // Two chunks of the same note; the note appears once, scored by its
            // stronger chunk.
            let idx = index(&[
                ("permethrin permethrin permethrin", "n1"),
                ("permethrin", "n1"),
                ("insulin", "n2"),
            ]);
            let hits = idx.search("permethrin", 5);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].note_id, "n1");
            assert_eq!(hits[0].chunk_id, "c0", "should report the stronger chunk");
        }

        #[test]
        fn empty_index_does_not_panic() {
            let idx = Bm25Index::new(&[], vec![], vec![], 1.5, 0.75);
            assert!(idx.is_empty());
            assert!(idx.search("anything", 5).is_empty());
        }

        #[test]
        fn empty_query_returns_nothing() {
            let idx = index(&[("permethrin", "n1")]);
            assert!(idx.search("", 5).is_empty());
        }

        #[test]
        fn shorter_document_outranks_longer_one_for_the_same_term() {
            // Length normalisation, the `b` parameter, is what does this.
            let padding = "filler ".repeat(60);
            let idx = index(&[("permethrin", "n1"), (&format!("permethrin {padding}"), "n2")]);
            let hits = idx.search("permethrin", 5);
            assert_eq!(hits[0].note_id, "n1");
        }
    }
}
