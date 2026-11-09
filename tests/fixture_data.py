"""Load the same JSON tensor data used to generate the C++ fixtures."""
import json
from pathlib import Path
import torch

FIXTURE_PATH = Path(__file__).resolve().parent / 'fixtures' / 'reference.json'

def load_fixtures(path=FIXTURE_PATH):
    document = json.loads(Path(path).read_text())
    tensors = {
        name: torch.tensor(item['data'], dtype=torch.float32).reshape(item['shape'])
        for name, item in document['tensors'].items()
    }
    return document, tensors
