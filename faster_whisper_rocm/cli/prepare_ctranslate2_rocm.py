"""Prepare ROCm CTranslate2 wheel for faster-whisper."""

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path


def _find_ctranslate2_pkg_dir() -> Path:
    """Locate the installed ctranslate2 package directory without importing it."""
    try:
        dist = importlib.metadata.distribution("ctranslate2")
    except importlib.metadata.PackageNotFoundError:
        print(
            "ERROR: ctranslate2 distribution not found in this environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Try to resolve the top-level package directory under site-packages.
    candidates = [
        Path(dist.locate_file("ctranslate2")),
        Path(dist.locate_file("ctranslate2/__init__.py")).parent,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()
    # Fallback: use base location and append
    base = Path(dist.locate_file(""))
    pkg = (base / "ctranslate2").resolve()
    if pkg.exists():
        return pkg
    print("ERROR: Could not determine ctranslate2 package directory.", file=sys.stderr)
    sys.exit(1)


def _check_patchelf() -> None:
    try:
        subprocess.run(
            ["patchelf", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "ERROR: patchelf not found. "
            "Please install patchelf (e.g., apt-get install patchelf, "
            "dnf install patchelf).",
            file=sys.stderr,
        )
        sys.exit(2)


def _copy_shared_lib(src_lib: Path, dst_dir: Path) -> Path:
    if not src_lib.is_file():
        print(f"ERROR: Source shared library not found: {src_lib}", file=sys.stderr)
        sys.exit(3)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src_lib.name
    shutil.copy2(src_lib, dst)
    return dst


def _ensure_soname_symlink(copied_lib: Path) -> None:
    """Create a SONAME symlink for the copied library, if needed.

    Many loaders resolve dependencies by SONAME (e.g., libctranslate2.so.3).
    If the file is versioned (e.g., libctranslate2.so.3.23.0), ensure a symlink
    libctranslate2.so.3 -> libctranslate2.so.3.23.0 exists in the same directory.
    """
    try:
        out = subprocess.check_output(
            ["patchelf", "--print-soname", str(copied_lib)], text=True
        ).strip()
        soname = out
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try to infer SONAME by trimming trailing version components
        name = copied_lib.name
        # If name looks like libfoo.so.X.Y[.Z], cut to libfoo.so.X
        parts = name.split(".so.")
        if len(parts) == 2 and parts[1] and parts[1][0].isdigit():
            major = parts[1].split(".")[0]
            soname = parts[0] + ".so." + major
        else:
            # Give up silently; many libs already have SONAME == filename
            return
    link_path = copied_lib.parent / soname
    if not link_path.exists():
        try:
            link_path.symlink_to(copied_lib.name)
        except FileExistsError:
            pass


def _patch_rpath_on_extensions(pkg_dir: Path, rel_lib_dir: str = ".rocm_libs") -> None:
    # Patch all ctranslate2 extension .so files to have RPATH=$ORIGIN/<rel_lib_dir>
    origin_rpath = f"$ORIGIN/{rel_lib_dir}"
    so_files = list(pkg_dir.glob("*_ext*.so")) + list(pkg_dir.glob("**/*_ext*.so"))
    if not so_files:
        # Fallback: patch any top-level .so under the package dir
        so_files = list(pkg_dir.glob("*.so"))
    if not so_files:
        print(f"ERROR: No extension .so files found under {pkg_dir}", file=sys.stderr)
        sys.exit(4)
    for so in so_files:
        subprocess.run(["patchelf", "--set-rpath", origin_rpath, str(so)], check=True)


def main(argv: list[str] | None = None) -> int:
    """Prepare and patch the CTranslate2 library for ROCm.

    This script locates the installed ctranslate2 package, copies the ROCm-enabled
    shared library into it, and patches the RPATH of the extension modules to find
    it. This allows the wheel to be self-contained without requiring LD_LIBRARY_PATH.

    Args:
        argv: Command-line arguments. If provided, the first argument is treated
            as the path to the source `libctranslate2.so` file.

    Returns:
        0 on success, non-zero on failure.
    """
    argv = argv or sys.argv[1:]
    # Allow overriding source lib path via CLI arg; defaults to:
    # 'out/ctranslate2_root/lib/libctranslate2.so.3'
    if argv:
        src_lib = Path(argv[0]).resolve()
    else:
        src_lib = (
            Path.cwd() / "out/ctranslate2_root/lib/libctranslate2.so.3"
        ).resolve()

    _check_patchelf()

    pkg_dir = _find_ctranslate2_pkg_dir()
    rocm_dir = pkg_dir / ".rocm_libs"
    copied = _copy_shared_lib(src_lib, rocm_dir)
    _ensure_soname_symlink(copied)

    _patch_rpath_on_extensions(pkg_dir, rel_lib_dir=rocm_dir.name)

    # Verify import without LD_LIBRARY_PATH by launching a clean subprocess
    code = (
        "import ctranslate2, sys; "
        "print('OK import ctranslate2', ctranslate2.__version__, 'from', "
        "ctranslate2.__file__)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        print(
            "ERROR: Verification import failed after patching:\n" + res.stderr,
            file=sys.stderr,
        )
        return 5
    print(res.stdout.strip())
    print(f"Patched RPATH and copied {copied} successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
