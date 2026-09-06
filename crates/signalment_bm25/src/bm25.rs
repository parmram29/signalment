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
//! 2. **Use the `1 +` form of idf, and floor it at zero.** The `1 +` inside the
//!    logarithm is not decoration: without it a term appearing in every
//!    document scores `ln(1) = 0` and the whole query returns nothing. With it,
//!    such a term still contributes a small positive amount (`ln(1.2)` for
//!    `N = 2, df = 2`), which is what the Python reference does. The
//!    `max(_, 0.0)` floor is therefore defensive rather than load-bearing here,
//!    but keep it: it guards the classic Robertson/Sparck-Jones variant, which
//!    can go negative, from silently subtracting score if anyone swaps the
//!    formula later.
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
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch.is_ascii_alphanumeric() {
            current_token.push(ch.to_ascii_lowercase());
        } else if ch == '-' || ch == '\'' {
            if !current_token.is_empty() {
                current_token.push(ch);
            }
        } else {
            if !current_token.is_empty() {
                tokens.push(current_token);
                current_token = String::new();
            }
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token);
    }

    tokens
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

        let n = texts.len() as f64;

        // Initialize data structures
        let mut tf = Vec::new();
        let mut lengths = Vec::new();

        // Process each document
        for text in texts {
            let tokens = tokenize(text);
            let mut term_freq = HashMap::new();

            for token in &tokens {
                *term_freq.entry(token.clone()).or_insert(0) += 1;
            }

            tf.push(term_freq);
            lengths.push(tokens.len());
        }

        // Calculate average length
        let avg_len = if !lengths.is_empty() {
            lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
        } else {
            0.0
        };

        // Calculate document frequency for each term
        let mut doc_freq: HashMap<String, u32> = HashMap::new();
        for doc_tf in &tf {
            for term in doc_tf.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Calculate IDF for each term
        let mut idf = HashMap::new();
        for (term, df) in doc_freq {
            let raw_idf = (1.0 + (n - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
            let idf_val = raw_idf.max(0.0); // Floor at 0.0 (pitfall 2)
            idf.insert(term, idf_val);
        }

        Bm25Index {
            note_ids,
            chunk_ids,
            tf,
            lengths,
            avg_len,
            idf,
            k1,
            b,
        }
    }

    /// Score `query` against every document, returning at most `k` hits,
    /// best first, deduplicated by note id.
    ///
    /// Documents scoring zero are omitted entirely — a non-match is not a
    /// result with a low score, it is not a result.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return vec![];
        }

        // Use 1.0 for avg_len when it is 0.0 (pitfall 3)
        let avg_len = if self.avg_len == 0.0 { 1.0 } else { self.avg_len };

        // Map to store scores for each document
        let mut doc_scores: Vec<(usize, f64)> = Vec::new();

        for (doc_idx, _) in self.note_ids.iter().enumerate() {
            let mut score = 0.0;

            // For each query token, calculate its contribution to the score
            for token in &query_tokens {
                // Get the term frequency in this document
                let tf_val = *self.tf[doc_idx].get(token).unwrap_or(&0) as f64;

                // Get the IDF for this term
                let idf_val = self.idf.get(token).copied().unwrap_or(0.0);

                // Skip if IDF is 0 (term not in index or in all documents)
                if idf_val == 0.0 {
                    continue;
                }

                // Calculate the score for this term in this document
                let numerator = tf_val * (self.k1 + 1.0);
                let denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (self.lengths[doc_idx] as f64) / avg_len);

                score += idf_val * (numerator / denominator);
            }

            // Only add documents with positive scores
            if score > 0.0 {
                doc_scores.push((doc_idx, score));
            }
        }

        // Group by note_id, keeping the highest-scoring chunk for each note (pitfall 4)
        let mut note_scores: HashMap<&String, (f64, usize)> = HashMap::new();
        for (doc_idx, doc_score) in doc_scores {
            let note_id = &self.note_ids[doc_idx];
            let entry = note_scores.entry(note_id).or_insert((0.0, doc_idx));

            // Update if this chunk has a higher score
            if doc_score > entry.0 {
                *entry = (doc_score, doc_idx);
            }
        }

        // Convert to hits
        let mut hits: Vec<Hit> = note_scores
            .into_iter()
            .map(|(note_id, (score, doc_idx))| Hit {
                note_id: note_id.clone(),
                score,
                chunk_id: self.chunk_ids[doc_idx].clone(),
            })
            .collect();

        // Sort by score descending, then by note_id ascending to ensure determinism
        hits.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.note_id.cmp(&b.note_id))
        });

        // Truncate to k results
        hits.truncate(k);

        hits
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
        fn ubiquitous_term_scores_low_but_not_zero() {
            // Verified against the Python reference: with the `1 +` form, a term
            // in every document has idf ln(1.2) = 0.1823, so it still matches
            // both documents equally rather than dropping out. Asserting it
            // returns nothing would be the classic-variant behaviour, which is
            // not what this crate implements.
            let idx = index(&[("cat insulin", "n1"), ("cat permethrin", "n2")]);
            let hits = idx.search("cat", 5);
            assert_eq!(hits.len(), 2);
            assert!((hits[0].score - hits[1].score).abs() < 1e-9, "equal evidence, equal score");
            assert!((hits[0].score - 0.1823).abs() < 1e-3, "got {}", hits[0].score);
        }

        #[test]
        fn rarer_term_outranks_commoner_one() {
            // What idf is actually for: "insulin" appears once, "cat" in both,
            // so a document matching the rare term must rank higher.
            let idx = index(&[("cat insulin", "n1"), ("cat permethrin", "n2")]);
            let hits = idx.search("insulin cat", 5);
            assert_eq!(hits[0].note_id, "n1");
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
