"""Sanity tests for the canonical crisis-event registry."""

from __future__ import annotations

import pytest

from regime.eval.crises import CRISES, all_names, by_name


def test_eight_events_registered():
    assert len(CRISES) == 8


def test_anchor_dates_are_ordered():
    """For each event: peak_date <= m5_date <= m10_date <= bottom_date."""
    for c in CRISES:
        assert c.peak_date <= c.m5_date, f"{c.name}: peak {c.peak_date} > m5 {c.m5_date}"
        assert c.m5_date <= c.m10_date, f"{c.name}: m5 {c.m5_date} > m10 {c.m10_date}"
        assert c.m10_date <= c.bottom_date, f"{c.name}: m10 {c.m10_date} > bottom {c.bottom_date}"


def test_unique_names():
    names = [c.name for c in CRISES]
    assert len(names) == len(set(names))


def test_by_name_lookup():
    c = by_name("Mar 2020 COVID")
    assert c.peak_date.year == 2020


def test_by_name_unknown_raises():
    with pytest.raises(KeyError):
        by_name("not a crisis")


def test_all_names_returns_in_order():
    names = all_names()
    assert names[0] == CRISES[0].name
    assert names[-1] == CRISES[-1].name
    assert len(names) == 8


def test_every_event_has_citation_note():
    for c in CRISES:
        assert c.note.strip(), f"{c.name} missing note"
        # Sanity: citation-ish content
        assert len(c.note) >= 20, f"{c.name} note too short"
