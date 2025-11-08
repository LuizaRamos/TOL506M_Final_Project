from pathlib import Path
import os, shutil
import kagglehub

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_data(force: bool = False):
    root = project_root()
    target_path = root / 'data' / 'animals10'
    target_path.mkdir(parents=True, exist_ok=True)

    # If already present and not forcing, just use it
    if target_path.exists() and not force:
        return target_path

    # Download latest version
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    print("Path to dataset files:", path)

    # Copy all files from downloaded path to target directory
    source_path = Path(path)
    for item in source_path.iterdir():
        if item.is_file():
            shutil.copy2(item, target_path / item.name)
        elif item.is_dir():
            shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)

    return target_path
