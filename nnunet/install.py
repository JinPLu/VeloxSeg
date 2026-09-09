from importlib.metadata import distribution
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent.parent
source = root / 'nnunet' / 'nnunetv2'
target = Path(distribution('nnunetv2').locate_file('nnunetv2'))
for source_file in source.rglob('*.py'):
    target_file = target / source_file.relative_to(source)
    if source_file.name == '__init__.py' and target_file.exists():
        continue
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_file, target_file)

# Link the public packages rather than copying another model or relying on a
# .pth file (Python skips files with a hidden flag on macOS).
for package in ('model', 'utils'):
    link = target.parent / package
    if link.is_symlink() and link.resolve() == root / package:
        continue
    if link.exists() or link.is_symlink():
        raise FileExistsError(f'{link} already belongs to another source; use a dedicated environment')
    link.symlink_to(root / package, target_is_directory=True)
(target.parent / 'veloxseg_source.pth').unlink(missing_ok=True)
