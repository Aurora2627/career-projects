import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
from run_phase2 import dhash
class PerceptualHashTests(unittest.TestCase):
    def test_decoded_content_not_file_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.png'; b=Path(tmp)/'b.bmp'
            im=Image.fromarray(np.tile(np.arange(90,dtype=np.uint8),(80,1)))
            im.save(a); im.save(b)
            self.assertEqual(dhash(a),dhash(b))
    def test_bucket_candidate_guarantee(self):
        h=0x123456789abcdef0
        for bit in range(64):
            modified=h ^ (1<<bit) ^ (1<<((bit+17)%64)) ^ (1<<((bit+35)%64))
            self.assertTrue(any(((h>>(16*c))&65535)==((modified>>(16*c))&65535) for c in range(4)))
    def test_direction_changes_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.png'; b=Path(tmp)/'b.png'
            x=np.tile(np.arange(90,dtype=np.uint8),(80,1))
            Image.fromarray(x).save(a); Image.fromarray(x[:,::-1]).save(b)
            self.assertEqual((dhash(a)^dhash(b)).bit_count(),64)
if __name__=='__main__': unittest.main()
