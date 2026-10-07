import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch_models import fit_tfidf,transform_tfidf,fit_standardizer,standardize,fit_softmax
from metrics import classification_metrics

class TorchModelTests(unittest.TestCase):
    def test_tfidf_unknown_terms_and_normalization(self):
        state=fit_tfidf(['red blue','red red'])
        features=transform_tfidf(['red blue','unknown'],state)
        self.assertAlmostEqual(float(features[0].norm()),1.,places=12)
        self.assertEqual(float(features[1].norm()),0.)
        red=state['vocabulary'].index('red');self.assertAlmostEqual(float(state['idf'][red]),1.)
    def test_train_only_scaling(self):
        state=fit_standardizer(torch.tensor([[0.,3.],[2.,3.]],dtype=torch.float64))
        heldout=standardize(torch.tensor([[100.,3.]],dtype=torch.float64),state)
        self.assertEqual(heldout.tolist(),[[99.,0.]])
    def test_weighted_optimization_and_reload(self):
        x=torch.eye(6,dtype=torch.float64).repeat(3,1);y=torch.arange(6).repeat(3)
        torch.set_num_threads(2);model,info=fit_softmax(x,y,1.,42)
        self.assertLess(info['objective'],1.8)
        self.assertEqual(model(x).argmax(1).tolist(),y.tolist())
        restored=torch.nn.Linear(6,6,dtype=torch.float64);restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
    def test_macro_metrics_and_tied_average_precision(self):
        y=torch.arange(6);scores=torch.ones(6,6)
        result=classification_metrics(y,y,scores)
        self.assertAlmostEqual(result['macro_f1'],1.)
        self.assertAlmostEqual(result['macro_average_precision'],1/6)
if __name__=='__main__':unittest.main()
