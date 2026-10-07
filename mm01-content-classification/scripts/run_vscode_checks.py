"""Run source compilation and project tests from VS Code's Python interpreter."""
import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    for folder in ['src', 'scripts', 'tests']:
        for path in (ROOT / folder).glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT / 'tests')))
    (ROOT/'reports/vscode-tests.json').write_text(json.dumps({
        'tests_run':result.testsRun, 'failures':len(result.failures),
        'errors':len(result.errors), 'skipped':len(result.skipped),
        'passed':result.wasSuccessful(), 'python':sys.executable,
        'launch_environment':'VS Code terminal/debugger'}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
