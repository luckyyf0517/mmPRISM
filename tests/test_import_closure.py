"""Every shipped Python module must import with accelerators hidden."""

import os
import subprocess
import sys
from pathlib import Path


def test_shipped_modules_import_on_cpu() -> None:
    root = Path(__file__).resolve().parents[1]
    program = """
import importlib
import ast
from pathlib import Path
import torch
assert not torch.cuda.is_available(), 'accelerators must be hidden for this check'
root = Path('src')
failures = []
modules = set()
for path in root.rglob('*.py'):
    name = '.'.join(path.relative_to(root).with_suffix('').parts)
    if name.endswith('.__init__'):
        name = name[:-len('.__init__')]
    modules.add(name)
for name in sorted(modules):
    try:
        importlib.import_module(name)
    except Exception as error:
        failures.append(f'{name}: {type(error).__name__}: {error}')
for path in root.rglob('*.py'):
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        if not node.module.startswith('mmprism.'):
            continue
        try:
            imported = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name != '*' and not hasattr(imported, alias.name):
                    importlib.import_module(node.module + '.' + alias.name)
        except Exception as error:
            failures.append(f'{path}:{node.lineno}: {type(error).__name__}: {error}')
assert not failures, '\\n'.join(failures)
print(f'{len(modules)} shipped modules imported; torch {torch.__version__}; CPU only')
"""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(root / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_data_and_scoring_import_without_training_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
import mmprism.data.compact_radar
import mmprism.release.data
import mmprism.release.scoring
import mmprism.release.verify
assert not {'torch', 'lightning', 'transformers'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1"),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
