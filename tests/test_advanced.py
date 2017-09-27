# -*- coding: utf-8 -*-

from .context import pos_ngrams

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(pos_ngrams.hmm())


if __name__ == '__main__':
    unittest.main()
