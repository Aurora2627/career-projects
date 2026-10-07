import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from data import normalize_text, check_split_leakage

class DataTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text(" A  B\n"),"a b")
    def test_text_leakage_rejected(self):
        with self.assertRaises(ValueError):
            check_split_leakage({"train":[{"text":"Hello","image_sha256":"a"}],"test":[{"text":" hello ","image_sha256":"b"}]})
    def test_image_leakage_rejected(self):
        with self.assertRaises(ValueError):
            check_split_leakage({"train":[{"text":"a","image_sha256":"same"}],"test":[{"text":"b","image_sha256":"same"}]})
    def test_clean_splits(self):
        check_split_leakage({"train":[{"text":"a","image_sha256":"a"}],"test":[{"text":"b","image_sha256":"b"}]})
if __name__=="__main__":unittest.main()
