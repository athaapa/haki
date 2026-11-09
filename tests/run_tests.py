"""Compile/run each test file separately, so one broken template doesn't hide other results.
Usage: python3 tests/run_tests.py [math|nn|rope|attention_head|multi_head|mlp|block|primitives]
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
sources = sorted(s for s in (root / 'tests').glob('test_*.cpp') if s.name != 'test_main.cpp')
args = sys.argv[1:]
sanitize = '--sanitize' in args
args = [a for a in args if a != '--sanitize']
if args:
    sources = [s for s in sources if s.stem == 'test_' + args[0]]
failed = 0
ran = 0
with tempfile.TemporaryDirectory(prefix='haki-tests-') as temp:
    for source in sources:
        names = re.findall(r'^void (test_\w+)\(\)', source.read_text(), re.M)
        if not names:
            continue
        ran += 1
        driver = Path(temp) / 'driver.cpp'
        driver.write_text('#include <iostream>\n#include <exception>\n' +
            '\n'.join(f'void {name}();' for name in names) + '\nint main() { int failures = 0;\n' +
            '\n'.join(f'try {{ {name}(); std::cout << "[PASS] {name}\\n"; }} '
                      f'catch (const std::exception& e) {{ ++failures; std::cerr << "[FAIL] {name}: " << e.what() << "\\n"; }}' for name in names) +
            '\nreturn failures ? 1 : 0; }\n')
        binary = Path(temp) / source.stem
        command = ['clang++', '-std=c++23', '-g', '-O0', '-ferror-limit=3', '-iquote', str(root / 'inc'),
                   str(source), str(driver), '-o', str(binary)]
        if sanitize:
            command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            failed += 1
            print(f'[BUILD FAIL] {source.name}', flush=True)
            print(result.stderr, flush=True)
            continue
        try:
            if subprocess.run([str(binary)], timeout=20).returncode:
                failed += 1
        except subprocess.TimeoutExpired:
            failed += 1
            print(f'[TIMEOUT] {source.name}', flush=True)
if not ran:
    sys.exit('No matching test group')
print(f'{ran} test groups checked; {failed} failed (including compilation failures).')
sys.exit(bool(failed))
