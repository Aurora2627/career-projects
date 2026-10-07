"""Check real gradients, model selection and checkpoint behavior on available devices."""
import unittest
import torch
from accelerated_training import train_head, load_head

class AcceleratedTrainingTests(unittest.TestCase):
    def check_device(self, device):
        x = torch.eye(6).repeat(4, 1)
        y = torch.arange(6).repeat(4)
        logs = []
        model, info = train_head(x, y, x, y, device, hidden=12,
                                 epochs=3, patience=3, lr=.05, log=logs.append)
        self.assertGreater(info['optimizer_steps'], 0)
        self.assertEqual(next(model.parameters()).device.type, device)
        self.assertTrue(all(r['dtype'] == 'torch.float32' for r in logs))
        self.assertEqual(info['val']['macro_f1'], max(r['val_macro_f1'] for r in logs))
        bundle = {'input_dim': 6, 'hidden_dim': 12, 'dtype': 'float32',
                  'state_dict': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        restored = load_head(bundle, device)
        with torch.inference_mode():
            torch.testing.assert_close(model(x.to(device)), restored(x.to(device)), rtol=0, atol=0)

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple MPS unavailable in this process')
    def test_mps(self):
        self.check_device('mps')
