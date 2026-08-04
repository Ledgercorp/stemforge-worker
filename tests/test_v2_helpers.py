from __future__ import annotations

import unittest
from app.daw_v2 import _vlq
from app.lyrics_v2 import preprocess_lyrics

class TestLyricPreprocessing(unittest.TestCase):
    def test_removes_section_labels_and_preserves_caps(self):
        result=preprocess_lyrics("[verse]\nLine one\n\n[breakdown]\nI AM HERE!")
        self.assertEqual(result["lines"],["Line one","I AM HERE!"])
        self.assertEqual(result["sections"][1]["name"],"breakdown")

class TestMidiEncoding(unittest.TestCase):
    def test_vlq(self):
        self.assertEqual(_vlq(0),b"\x00")
        self.assertEqual(_vlq(127),b"\x7f")
        self.assertEqual(_vlq(128),b"\x81\x00")

if __name__=="__main__":unittest.main()
