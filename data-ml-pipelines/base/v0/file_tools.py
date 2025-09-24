import os
import shutil
import tempfile
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import List, Union

PathStr = Union[Path, str]


@contextmanager
def temp_open_file(path: PathStr, mode: str):
    file = open(path, mode)

    try:
        yield file
    finally:
        file.close()
        os.remove(path)


@contextmanager
def temp_dir(base: PathStr = None, paths: List[PathStr] = None):
    dir_path: Path = Path(tempfile.mkdtemp(suffix=None, prefix=None, dir=base))

    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)


def collect_files(
        base: Path,
        src_dirs: List[PathStr] = None,
        src_files: List[PathStr] = None,
        only_py: bool = True
) -> list[str]:
    assert src_dirs or src_files, "dirs of files must be provided"
    results = (src_files or []) and [base / path for path in chain(src_files, src_dirs)]

    pattern: str = (only_py and "**/*.py") or "**/*"
    for src_dir in map(Path, src_dirs):
        if not src_dir.is_absolute():
            src_dir = base / src_dir
        results.extend([file_path for file_path in src_dir.rglob(pattern)])

    return results


@contextmanager
def collect_folders(
    root: Path,
    paths: List[str]
) -> list[str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dst = Path(temp_dir)
        for p in paths:
            dst_dir = temp_dst / p
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(root / p, dst_dir, dirs_exist_ok=True)
        yield temp_dst
