"""Note review freshness.

The point is that the corpus tells you when it is going out of date, rather
than you having to remember. `last_reviewed` is the signal; a note that has
never been reviewed falls back to `authored`, which is the honest lower bound
on how long the content has gone unchecked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .schema import Note

# Notes carrying clinical numbers age worse than conceptual ones: a mechanism
# note is as true as it was last year, a protocol may not be.
CLINICAL_TYPES = frozenset({"dosage", "protocol"})


@dataclass(frozen=True)
class StaleNote:
    note: Note
    days: int
    never_reviewed: bool

    @property
    def is_clinical(self) -> bool:
        return self.note.note_type in CLINICAL_TYPES

    @property
    def basis(self) -> str:
        return "authored" if self.never_reviewed else "reviewed"


def stale_notes(
    notes: list[Note],
    max_age_days: int,
    today: date | None = None,
) -> list[StaleNote]:
    """Notes unreviewed for longer than max_age_days, most urgent first.

    Clinical notes sort ahead of everything else at equal staleness, so a
    long report never buries the dosage note that needs attention behind
    fifty definitions.
    """
    if max_age_days < 0:
        raise ValueError("max_age_days must not be negative")
    today = today or date.today()

    stale: list[StaleNote] = []
    for note in notes:
        checked = note.last_reviewed or note.authored
        days = (today - checked).days
        if days > max_age_days:
            stale.append(
                StaleNote(note=note, days=days, never_reviewed=note.last_reviewed is None)
            )
    return sorted(stale, key=lambda s: (not s.is_clinical, -s.days, s.note.id))
