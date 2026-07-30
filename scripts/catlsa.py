#!/usr/bin/env python3
"""catlsa — transparent cat/ls wrapper with archive support.

Walks a path left-to-right, resolving archives (tar, zip, 7z, gz, bz2, xz)
and auto-dispatching the leaf: file -> cat, directory -> listing.
Only the necessary files are extracted, not the entire archive.
Read-only — archives are never modified. Use vima.sh for editing.
Leaf reads and nested-archive descent are zero-disk for all formats except 7z.

usage: catlsa <path>

examples:
  catlsa project.tar.gz/src/main.py          # cat the file
  catlsa project.tar.gz/src/                 # list the directory (no extraction)
  catlsa outer.zip/inner.tar.gz/config.yml   # nested archives
  catlsa data.tar.gz                         # list root of archive
"""

import os
import sys
import io
import gzip
import bz2
import lzma
import tarfile
import zipfile
import shutil
import atexit
import tempfile
import subprocess
from typing import Optional

MAX_NESTING = 16
TEMPDIRS: list[str] = []


def cleanup() -> None:
    for d in TEMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(cleanup)


def _stem(filepath: str, suffix: str) -> str:
    name = os.path.basename(filepath)
    if name.endswith("." + suffix):
        return name[: -(len(suffix) + 1)]
    return name


# ---- format detection (two variants: fs via `file`, memory via magic bytes) ----


def detect_format_fs(filepath: str) -> Optional[str]:
    """Detect archive format via `file --mime-type`."""
    if not os.path.isfile(filepath):
        return None
    try:
        mime = subprocess.run(
            ["file", "-b", "--mime-type", filepath],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None

    if mime == "application/zip":
        return "zip"
    if mime == "application/x-tar":
        return "tar"
    if mime == "application/x-7z-compressed":
        return "7z"
    if mime == "application/gzip":
        inner = subprocess.run(
            ["file", "-zb", filepath], capture_output=True, text=True
        ).stdout.strip()
        return "tar.gz" if "tar" in inner else "gz"
    if mime == "application/x-bzip2":
        inner = subprocess.run(
            ["file", "-zb", filepath], capture_output=True, text=True
        ).stdout.strip()
        return "tar.bz2" if "tar" in inner else "bz2"
    if mime == "application/x-xz":
        inner = subprocess.run(
            ["file", "-zb", filepath], capture_output=True, text=True
        ).stdout.strip()
        return "tar.xz" if "tar" in inner else "xz"

    return None


# Magic bytes
_SIG_ZIP = b"PK\x03\x04"
_SIG_GZ = b"\x1f\x8b"
_SIG_BZ2 = b"BZh"
_SIG_XZ = b"\xfd7zXZ\x00"
_SIG_7Z = b"7z\xbc\xaf'\x1c"
_TAR_MAGIC = b"ustar"


def _has_tar_magic(data: bytes) -> bool:
    return len(data) >= 262 and data[257:262] == _TAR_MAGIC


def detect_format_bytes(data: bytes, filename: str = "") -> Optional[str]:
    """Detect archive format from magic bytes + optional filename hint."""
    if data[:4] == _SIG_ZIP:
        return "zip"
    if _has_tar_magic(data):
        return "tar"
    if data[:2] == _SIG_GZ:
        try:
            head = gzip.decompress(data[:4096])
        except Exception:
            head = b""
        return "tar.gz" if _has_tar_magic(head) else "gz"
    if data[:3] == _SIG_BZ2:
        try:
            head = bz2.decompress(data[:4096])
        except Exception:
            head = b""
        return "tar.bz2" if _has_tar_magic(head) else "bz2"
    if data[:6] == _SIG_XZ:
        try:
            head = lzma.decompress(data[:4096])
        except Exception:
            head = b""
        return "tar.xz" if _has_tar_magic(head) else "xz"
    if data[:6] == _SIG_7Z:
        return "7z"

    # Fallback to filename extension if magic is ambiguous or missing
    name = filename.lower()
    suffices = [
        (".tar.gz", "tar.gz"), (".tgz", "tar.gz"),
        (".tar.bz2", "tar.bz2"), (".tbz2", "tar.bz2"),
        (".tar.xz", "tar.xz"), (".txz", "tar.xz"),
        (".tar", "tar"),
        (".zip", "zip"),
        (".gz", "gz"),
        (".bz2", "bz2"),
        (".xz", "xz"),
        (".7z", "7z"),
    ]
    for suffix, fmt in suffices:
        if name.endswith(suffix):
            return fmt

    return None


# ---- tar helpers (handle ./ prefix variations) ----


def _tar_find(tf: tarfile.TarFile, path: str):
    """Find a tar member by path. Returns TarInfo, None (directory prefix), or raises KeyError."""
    for candidate in (path, "./" + path, path + "/", "./" + path + "/"):
        try:
            return tf.getmember(candidate)
        except KeyError:
            continue
    for member in tf.getmembers():
        name = member.name.rstrip("/")
        if name == path or name == "./" + path:
            return member
        if name.startswith(path + "/") or name.startswith("./" + path + "/"):
            return None
    raise KeyError(path)


# ---- in-memory archive descent (zero-disk except 7z) ----


def _descend_tar(tf: tarfile.TarFile, inner_path: str, depth: int) -> None:
    """Walk inside a tar archive. tf may be on-disk or BytesIO-backed."""
    if depth > MAX_NESTING:
        sys.exit("error: max nesting depth exceeded")

    if not inner_path:
        for m in tf.getmembers():
            print(m.name)
        return

    r = inner_path
    current = ""
    while r:
        seg, _, remainder = r.partition("/")
        current = f"{current}/{seg}" if current else seg
        is_last = not remainder

        try:
            member = _tar_find(tf, current)
        except KeyError:
            sys.exit(f"error: not found in archive: {current}")

        is_dir = member is None or member.isdir()

        if is_last:
            if is_dir:
                prefix = current if current.endswith("/") else current + "/"
                for m in tf.getmembers():
                    if m.name.startswith(prefix) or m.name.startswith("./" + prefix):
                        print(m.name)
            else:
                data = tf.extractfile(member).read()
                inner_fmt = detect_format_bytes(data, current)
                if inner_fmt:
                    _descend_bytes(data, inner_fmt, "", depth + 1)
                else:
                    sys.stdout.buffer.write(data)
            return

        if is_dir:
            r = remainder
            continue

        # File, not last -> nested archive
        data = tf.extractfile(member).read()
        inner_fmt = detect_format_bytes(data, current)
        if not inner_fmt:
            sys.exit(f"error: not an archive: {current}")
        _descend_bytes(data, inner_fmt, remainder, depth + 1)
        return


def _descend_zip(zf: zipfile.ZipFile, inner_path: str, depth: int) -> None:
    """Walk inside a zip archive. zf may be on-disk or BytesIO-backed."""
    if depth > MAX_NESTING:
        sys.exit("error: max nesting depth exceeded")

    if not inner_path:
        for info in zf.infolist():
            print(info.filename)
        return

    r = inner_path
    current = ""
    while r:
        seg, _, remainder = r.partition("/")
        current = f"{current}/{seg}" if current else seg
        is_last = not remainder

        # Resolve current in zip
        try:
            info = zf.getinfo(current)
            is_dir = info.is_dir()
        except KeyError:
            try:
                zf.getinfo(current + "/")
                is_dir = True
            except KeyError:
                # Check if any entry lives under this prefix
                if any(
                    i.filename.startswith(current + "/")
                    for i in zf.infolist()
                ):
                    is_dir = True
                else:
                    sys.exit(f"error: not found in archive: {current}")

        if is_last:
            if is_dir:
                prefix = current + "/"
                for info in zf.infolist():
                    if info.filename.startswith(prefix):
                        print(info.filename)
            else:
                data = zf.read(current)
                inner_fmt = detect_format_bytes(data, current)
                if inner_fmt:
                    _descend_bytes(data, inner_fmt, "", depth + 1)
                else:
                    sys.stdout.buffer.write(data)
            return

        if is_dir:
            r = remainder
            continue

        # File, not last -> nested archive
        data = zf.read(current)
        inner_fmt = detect_format_bytes(data, current)
        if not inner_fmt:
            sys.exit(f"error: not an archive: {current}")
        _descend_bytes(data, inner_fmt, remainder, depth + 1)
        return


def _descend_bytes(data: bytes, fmt: str, inner_path: str, depth: int) -> None:
    """Descent into an in-memory archive."""
    if fmt in ("tar",):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            _descend_tar(tf, inner_path, depth)
    elif fmt in ("tar.gz", "tgz"):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            _descend_tar(tf, inner_path, depth)
    elif fmt in ("tar.bz2", "tbz2"):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tf:
            _descend_tar(tf, inner_path, depth)
    elif fmt in ("tar.xz", "txz"):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tf:
            _descend_tar(tf, inner_path, depth)
    elif fmt == "zip":
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            _descend_zip(zf, inner_path, depth)
    elif fmt == "gz":
        dec = gzip.decompress(data)
        inner2 = detect_format_bytes(dec, "")
        if inner2:
            _descend_bytes(dec, inner2, inner_path, depth)
        else:
            if inner_path:
                sys.exit("error: not an archive")
            sys.stdout.buffer.write(dec)
    elif fmt == "bz2":
        dec = bz2.decompress(data)
        inner2 = detect_format_bytes(dec, "")
        if inner2:
            _descend_bytes(dec, inner2, inner_path, depth)
        else:
            if inner_path:
                sys.exit("error: not an archive")
            sys.stdout.buffer.write(dec)
    elif fmt == "xz":
        dec = lzma.decompress(data)
        inner2 = detect_format_bytes(dec, "")
        if inner2:
            _descend_bytes(dec, inner2, inner_path, depth)
        else:
            if inner_path:
                sys.exit("error: not an archive")
            sys.stdout.buffer.write(dec)
    elif fmt == "7z":
        # 7z from bytes -> must write to temp file, then use fs path
        tmpdir = tempfile.mkdtemp(prefix="catlsa.")
        TEMPDIRS.append(tmpdir)
        tmpfile = os.path.join(tmpdir, "archive.7z")
        with open(tmpfile, "wb") as f:
            f.write(data)
        _walk_fs_archive(tmpfile, "7z", inner_path, depth)


# ---- filesystem archive entry points ----


def _walk_fs_archive(archive_path: str, fmt: str, inner_path: str, depth: int) -> None:
    """Open an archive on the filesystem and walk into it."""
    if fmt in ("tar", "tar.gz", "tar.bz2", "tar.xz"):
        with tarfile.open(archive_path, "r:*") as tf:
            _descend_tar(tf, inner_path, depth)
    elif fmt == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            _descend_zip(zf, inner_path, depth)
    elif fmt in ("gz", "bz2", "xz"):
        openers = {"gz": gzip.open, "bz2": bz2.open, "xz": lzma.open}
        with openers[fmt](archive_path, "rb") as f:
            data = f.read()
        inner_fmt = detect_format_bytes(data, archive_path)
        if inner_fmt:
            _descend_bytes(data, inner_fmt, inner_path, depth)
        else:
            if inner_path:
                sys.exit(f"error: not an archive: {archive_path}")
            sys.stdout.buffer.write(data)
    elif fmt == "7z":
        # 7z always needs disk
        if not inner_path:
            subprocess.run(["7z", "l", archive_path])
            return
        _descend_7z(archive_path, inner_path, depth)


def _descend_7z(archive_path: str, inner_path: str, depth: int) -> None:
    """Walk inside a 7z archive (disk-backed — no Python stdlib for 7z)."""
    if depth > MAX_NESTING:
        sys.exit("error: max nesting depth exceeded")

    tmpdir = tempfile.mkdtemp(prefix="catlsa.")
    TEMPDIRS.append(tmpdir)

    r = inner_path
    current = ""
    while r:
        seg, _, remainder = r.partition("/")
        current = f"{current}/{seg}" if current else seg
        is_last = not remainder

        # Stat: try as file, then as directory prefix
        r1 = subprocess.run(
            ["7z", "l", archive_path, current],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            r1 = subprocess.run(
                ["7z", "l", archive_path, current + "/"],
                capture_output=True, text=True,
            )
            if r1.returncode != 0:
                sys.exit(f"error: not found in archive: {current}")
            is_dir = True
        else:
            is_dir = current.endswith("/") or not r1.stdout.strip()

        if is_last:
            if is_dir:
                subprocess.run(["7z", "l", archive_path, current])
            else:
                subprocess.run(
                    ["7z", "x", f"-o{tmpdir}", archive_path, current],
                    capture_output=True, check=True,
                )
                target = os.path.join(tmpdir, current)
                with open(target, "rb") as f:
                    sys.stdout.buffer.write(f.read())
            return

        if is_dir:
            r = remainder
            continue

        # File, not last -> nested archive (must extract to disk for 7z)
        subprocess.run(
            ["7z", "x", f"-o{tmpdir}", archive_path, current],
            capture_output=True, check=True,
        )
        target = os.path.join(tmpdir, current)
        fmt2 = detect_format_fs(target)
        if not fmt2:
            sys.exit(f"error: not an archive: {current}")
        _walk_fs_archive(target, fmt2, remainder, depth + 1)
        return


# ---- main walk entry (filesystem only — finds first archive, then delegates) ----


def walk(root: str, path: str, depth: int = 0) -> None:
    if path.startswith("/"):
        root = "/"
        path = path.lstrip("/")

    if depth > MAX_NESTING:
        sys.exit("error: max nesting depth exceeded")

    prefix = ""
    inner = ""
    archive = None
    rest = path

    while rest:
        seg, _, remainder = rest.partition("/")
        prefix = f"{prefix}/{seg}" if prefix else seg
        candidate = os.path.join(root, prefix) if root != "/" else "/" + prefix
        rest = remainder or ""

        if os.path.isfile(candidate):
            fmt = detect_format_fs(candidate)
            if fmt:
                archive = os.path.realpath(candidate)
                inner = rest
                break

    if archive is None:
        target = os.path.join(root, path) if root != "/" else "/" + path
        if os.path.isdir(target):
            subprocess.run(["ls", "-lA", target])
            return
        if os.path.isfile(target):
            with open(target, "rb") as f:
                sys.stdout.buffer.write(f.read())
            return
        sys.exit(f"error: not found: {path}")

    fmt = detect_format_fs(archive)
    if not fmt:
        sys.exit(f"error: cannot detect format: {archive}")

    _walk_fs_archive(archive, fmt, inner, depth)


# ---- entry point ----


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 else 1)

    input_path = sys.argv[1]
    if input_path.startswith("./"):
        input_path = input_path[2:]

    if not shutil.which("file"):
        sys.exit("error: 'file' command not found")

    walk(".", input_path)


if __name__ == "__main__":
    main()
