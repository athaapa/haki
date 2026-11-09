"""Convert shared JSON fixtures to the C++ header (no PyTorch required)."""
import json
import math
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'

def generate_header(source=FIXTURE_DIR / 'reference.json', destination=FIXTURE_DIR / 'reference.h'):
    document = json.loads(source.read_text())
    lines = ['// Generated from reference.json by tests/fixtures_to_header.py.',
             '#pragma once', '#include <vector>', 'namespace fixtures {']
    for name, tensor in document['tensors'].items():
        values = tensor['data']
        if math.prod(tensor['shape']) != len(values):
            raise ValueError(f'{name}: shape does not match data length')
        if tensor['dtype'] != 'float32' or not all(math.isfinite(v) for v in values):
            raise ValueError(f'{name}: expected finite float32 fixture')
        lines.append('inline const std::vector<float> ' + name + ' = {' +
                     ', '.join(f'{v:.9e}f' for v in values) + '};')
    lines.append('}')
    destination.write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    generate_header()
