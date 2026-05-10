"""Unit tests for the data.lock manifest."""

from __future__ import annotations

from datetime import date

from regime.data import manifest


def test_lock_roundtrip(tmp_data_root, synthetic_ticker_factory, monkeypatch):
    prices = [(date(2020, 1, d), 100.0) for d in range(1, 11)]
    synthetic_ticker_factory("ABC", prices)

    # Place data.lock alongside data root for the test
    monkeypatch.chdir(tmp_data_root.parent)
    p = manifest.write_lock(tmp_data_root)
    assert p.exists()
    assert manifest.verify_lock(tmp_data_root)


def test_lock_detects_drift(tmp_data_root, synthetic_ticker_factory, monkeypatch):
    prices = [(date(2020, 1, d), 100.0) for d in range(1, 11)]
    synthetic_ticker_factory("ABC", prices)

    monkeypatch.chdir(tmp_data_root.parent)
    manifest.write_lock(tmp_data_root)
    assert manifest.verify_lock(tmp_data_root)

    # Mutate a partition: rebuild with different prices → SHA changes
    prices2 = [(date(2020, 1, d), 999.0) for d in range(1, 11)]
    synthetic_ticker_factory("ABC", prices2)
    assert not manifest.verify_lock(tmp_data_root)


def test_lock_missing_returns_false(tmp_data_root, monkeypatch):
    monkeypatch.chdir(tmp_data_root.parent)
    assert not manifest.verify_lock(tmp_data_root)
