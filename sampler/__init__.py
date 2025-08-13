# sampler/__init__.py

try:
    from . import _rs_sampler
except ImportError as e:
    # If the import fails, we try to build the Rust extension
    import os
    import shutil
    import subprocess
    import sys
    import sysconfig

    env = os.environ.copy()
    env['PYO3_PYTHON'] = sys.executable
    # Run cargo build --release in the sampler directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cargo_command = ['cargo', 'build', '--release']
    os_type = os.name

    if shutil.which('cargo') is None:
        raise ImportError(
            'Cargo is not installed or not found in PATH. '
            'Unable to build the Rust extension.'
        ) from e
    subprocess.run(cargo_command, cwd=current_dir, env=env, check=True)

    # Move the built .dylib to the current directory
    possible_exts = ['.dylib', '.so', '.pyd', '.dll']
    src_dir = os.path.join(current_dir, 'target', 'release')
    src_name = [
        f for f in os.listdir(src_dir)
        if any(f.endswith(ext) for ext in possible_exts)
    ]

    target_ext = sysconfig.get_config_var('EXT_SUFFIX')
    target_dir = current_dir
    # Handle target name: 1. remove trailing extension, 2. remove 'lib' prefix
    def clean_target_name(name):
        for ext in possible_exts:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break
        if name.startswith('lib'):
            name = name[3:]
        return name
    target_name = [
        clean_target_name(f) + target_ext
        for f in src_name
    ]

    for src, target in zip(src_name, target_name):
        src_path = os.path.join(src_dir, src)
        target_path = os.path.join(target_dir, target)
        shutil.move(src_path, target_path)

    subprocess.run(['cargo', 'clean'], cwd=current_dir, env=env, check=True)

_ensure_compile = 0
