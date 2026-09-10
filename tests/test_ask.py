"""The ask command: the product surface.

Two behaviours matter more than the formatting: it answers what the corpus
covers, and it declines what the corpus does not. The second is the one worth
guarding, because a system that bluffs is worse than no system.
"""

from __future__ import annotations

from signalment.cli import main


def run(capsys, *args) -> str:
    code = main(["ask", *args])
    assert code == 0
    return capsys.readouterr().out


class TestAnswering:
    def test_finds_the_note_that_answers(self, capsys):
        out = run(capsys, "why do newborn kittens get low blood sugar", "--species", "feline")
        assert "vet-endo-0142" in out

    def test_shows_review_state_with_every_citation(self, capsys):
        # A citation without its confidence invites trusting an unchecked note.
        out = run(capsys, "permethrin sodium channels")
        assert "verified" in out
        assert "Sources:" in out

    def test_warns_when_a_cited_note_is_unverified(self, capsys):
        out = run(capsys, "insulinoma episodic collapse", "--species", "canine")
        assert "reviewed" in out
        assert "not confirmed" in out


class TestSpeciesFilter:
    def test_same_question_different_species_different_answer(self, capsys):
        feline = run(capsys, "hypoglycaemia", "--species", "feline", "-k", "1")
        canine = run(capsys, "hypoglycaemia", "--species", "canine", "-k", "1")
        assert "vet-endo-0142" in feline and "vet-endo-0143" not in feline
        assert "vet-endo-0143" in canine and "vet-endo-0142" not in canine


class TestAbstention:
    def test_declines_a_question_the_corpus_cannot_answer(self, capsys):
        # Absolute BM25 scores cannot separate this from a real answer — both
        # peak near 3. The margin can, which is why the gate is relative.
        out = run(capsys, "what is the surgical repair for a cruciate ligament rupture")
        assert "Not covered by your notes" in out

    def test_declining_names_the_nearest_notes(self, capsys):
        out = run(capsys, "what is the surgical repair for a cruciate ligament rupture")
        assert "Nearest" in out

    def test_margin_check_can_be_disabled(self, capsys):
        # Disabling it should let the same query through, which is what proves
        # the margin is what declined it rather than the absolute threshold.
        out = run(
            capsys,
            "what is the surgical repair for a cruciate ligament rupture",
            "--min-margin",
            "0",
        )
        assert "Not covered" not in out

    def test_nonsense_returns_nothing_at_all(self, capsys):
        out = run(capsys, "zzzqqq nonexistent terminology")
        assert "Not covered by your notes" in out


class TestGuards:
    def test_empty_corpus_is_reported(self, tmp_path, capsys):
        (tmp_path / "README.md").write_text("no notes here")
        assert main(["ask", "anything", "--corpus", str(tmp_path)]) == 1
        assert "corpus is empty" in capsys.readouterr().err
