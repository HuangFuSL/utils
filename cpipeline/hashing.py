import enum
import hashlib
from pathlib import Path
from typing import Dict, Iterable


class HashPolicy(enum.Enum):
    MTIME = 'mtime'
    SAMPLE = 'sample'
    FULL = 'full'


_SAMPLE_SIZE = 8192

def hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def hash_file(path: Path, policy: HashPolicy) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f'File not found: {path}')

    if policy == HashPolicy.MTIME:
        st = path.stat()
        return {str(path): hashlib.sha256(
            f'{st.st_mtime_ns}:{st.st_size}'.encode()
        ).hexdigest()}

    size = path.stat().st_size

    if policy == HashPolicy.SAMPLE and size > _SAMPLE_SIZE * 4:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            h.update(f.read(_SAMPLE_SIZE))
            f.seek(size // 2 - _SAMPLE_SIZE // 2)
            h.update(f.read(_SAMPLE_SIZE))
            f.seek(max(size - _SAMPLE_SIZE, 0))
            h.update(f.read(_SAMPLE_SIZE))
        return {str(path): h.hexdigest()}

    # FULL (fallback for SAMPLE on small files)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return {str(path): h.hexdigest()}


def _glob(base: Path, pattern: str = '**/*') -> list[Path]:
    return sorted(
        filter(lambda f: f.is_file(), base.glob(pattern)),
        key=lambda f: str(f.relative_to(base))
    )

def hash_dir(path: Path, policy: HashPolicy) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for f in _glob(path):
        result.update(hash_file(f, policy))
    return result


def hash_glob(pattern: str, base: Path, policy: HashPolicy) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for f in _glob(base, pattern):
        result.update(hash_file(f, policy))
    return result


def hash_entry(
    entry: str | Iterable[str | Path] | Path, base_path: Path, hash_policy: HashPolicy
) -> Dict[str, str]:
    if isinstance(entry, (str, Path)):
        entry = [entry]
    ret = {}
    for e in entry:
        # Glob
        if isinstance(e, str) and any(c in e for c in '*?['):
            hashes = hash_glob(e, base_path, hash_policy)
        else:
            path = base_path / e
            if not path.exists():
                raise FileNotFoundError(f'Path not found: {path}')
            # Directory
            if path.is_dir():
                hashes = hash_dir(path, hash_policy)
            # File
            else:
                hashes = hash_file(path, hash_policy)
        ret[e] = hash_string(''.join(
            map(lambda x: x[1], sorted(hashes.items(), key=lambda x: x[0]))
        ))
    return ret
