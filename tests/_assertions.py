"""Shared assertion helpers for indicator tests."""

from __future__ import annotations

import unittest


class IndicatorAssertions(unittest.TestCase):
    def assert_values_equal(self, actual, expected) -> None:
        self.assertEqual(len(actual), len(expected))
        for index, (got, want) in enumerate(zip(actual, expected)):
            with self.subTest(index=index):
                if want is None:
                    self.assertIsNone(got)
                else:
                    self.assertIsNotNone(got)
                    self.assertAlmostEqual(got, want, places=10)
