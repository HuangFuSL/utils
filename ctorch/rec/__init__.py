import os
import sys

if sys.platform != "linux":
    raise NotImplementedError("torchrec is only supported on Linux")

_BUILT_PATH = None


def _build_cdylib():
    """Build the LMDB PS plugin cdylib via cargo."""
    global _BUILT_PATH
    if _BUILT_PATH is not None:
        return _BUILT_PATH

    import shutil
    import subprocess

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cargo_bin = shutil.which('cargo')
    if cargo_bin is None:
        raise RuntimeError('Cargo is not installed or not found in PATH.')

    env = os.environ.copy()
    env['PYO3_PYTHON'] = sys.executable

    subprocess.run(
        [cargo_bin, 'build', '--release'],
        cwd=current_dir, env=env, check=True,
    )

    possible_exts = ('.so', '.dylib', '.dll')
    src_dir = os.path.join(current_dir, 'target', 'release')
    src_files = [
        f for f in os.listdir(src_dir)
        if any(f.endswith(ext) for ext in possible_exts)
    ]

    if not src_files:
        raise FileNotFoundError(
            f"No built cdylib found in {src_dir}"
        )

    target_name = src_files[0]
    target_path = os.path.join(current_dir, target_name)

    if not os.path.exists(target_path) or (
        os.path.getmtime(os.path.join(src_dir, target_name))
        > os.path.getmtime(target_path)
    ):
        shutil.copy2(
            os.path.join(src_dir, target_name), target_path,
        )

    _BUILT_PATH = target_path
    return target_path


def _find_library_path():
    """Return the absolute path to the compiled shared library."""
    if _BUILT_PATH is not None and os.path.exists(_BUILT_PATH):
        return _BUILT_PATH

    current_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = ('liblmdb_ps*.so', 'liblmdb_ps*.dylib', 'lmdb_ps*.dll')
    import glob
    for pattern in patterns:
        candidates = glob.glob(os.path.join(current_dir, pattern))
        if candidates:
            return candidates[0]

    return _build_cdylib()


def register_lmdb_io(path=None):
    """Register the LMDB parameter-server backend with TorchRec.

    After calling this function, TorchRec's dynamic embedding can use the
    ``"lmdb://path=/data/emb&map_size=50G"`` URL scheme.

    Args:
        path: Optional path to the shared library.  If not given, the
              auto-built library in this package is used.
    """
    import torch

    if path is None:
        path = _find_library_path()

    torch.ops.tde.register_io(path)
