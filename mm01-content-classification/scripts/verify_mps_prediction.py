"""Re-extract a real test image/text and validate the saved fusion checkpoint."""
import json
import torch
from predict import ROOT, predict

if __name__ == '__main__':
    torch.set_num_threads(4)
    run = ROOT/'runs/vscode-mps-v1'
    row = json.loads((run/'input_manifests/test.jsonl').read_text().splitlines()[0])
    expected = json.loads((run/'fusion-linear-predictions.jsonl').read_text().splitlines()[0])
    actual = predict('runs/vscode-mps-v1/fusion-linear.pt', ROOT/row['image'], row['text'], 'mps')
    difference = max(abs(a-b) for a,b in zip(actual['probabilities'], expected['probabilities']))
    assert actual['label'] == expected['prediction']
    assert difference < 1e-4
    assert actual['encoder_device'] == actual['classifier_device'] == 'mps'
    result = {'single_image_text_prediction':'passed', 'maximum_probability_difference':difference,
              'sample_id':row['id'], 'prediction':actual,
              'launch_environment':'VS Code integrated terminal'}
    (ROOT/'reports/vscode-mps-prediction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
