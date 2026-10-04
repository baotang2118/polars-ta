import unittest

from polars_ta import hello


class TestHello(unittest.TestCase):
    def test_hello_returns_greeting(self) -> None:
        self.assertEqual(hello(), "Hello, world!")
