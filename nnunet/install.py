from importlib.metadata import distribution
from pathlib import Path
import shutil

source = Path(__file__).resolve().parent / 'nnunetv2'
target = Path(distribution('nnunetv2').locate_file('nnunetv2'))
for source_file in source.rglob('*.py'):
    target_file = target / source_file.relative_to(source)
    if source_file.name == '__init__.py' and target_file.exists():
        continue
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_file, target_file)
