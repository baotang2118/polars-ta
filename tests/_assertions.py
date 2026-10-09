"""Shared assertion helpers for indicator tests."""

from __future__ import annotations

from pytest import approx

# The tolerance the suite's former assertAlmostEqual(places=10) accepted.
TOLERANCE = 5e-11


def assert_values_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        if want is None:
            assert got is None, f"index {index}"
        else:
            assert got is not None, f"index {index}"
            assert got == approx(want, rel=0, abs=TOLERANCE), f"index {index}"
