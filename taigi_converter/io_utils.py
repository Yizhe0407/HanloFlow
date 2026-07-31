from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
    import msvcrt
else:  # pragma: no cover - branch selection is platform-specific
    import fcntl


def _try_lock(fd: int) -> bool:
    if os.name == "nt":
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(fd, fcntl.LOCK_UN)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("檔案寫入未取得進展")
        view = view[written:]


@contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    timeout: float = 30.0,
    mode: int = 0o644,
) -> Iterator[None]:
    """Acquire an OS-managed cross-process advisory lock.

    The lock file is intentionally persistent. Keeping one inode avoids the
    unlink/recreate race inherent in ``O_EXCL`` lock files, while the operating
    system automatically releases the lock if a process exits or crashes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, mode)
    acquired = False
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        else:  # pragma: no cover - Windows
            os.chmod(path, mode)

        # Windows byte-range locking requires a byte to exist. This is harmless
        # on POSIX and happens before any process can depend on lock metadata.
        if os.fstat(fd).st_size == 0:
            _write_all(fd, b"\0")
            os.fsync(fd)

        deadline = time.monotonic() + timeout
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待檔案鎖逾時: {path}")
            time.sleep(0.05)
        acquired = True

        owner = json.dumps(
            {
                "token": uuid.uuid4().hex,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired": time.time(),
            },
            separators=(",", ":"),
        ).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        _write_all(fd, owner)
        os.fsync(fd)
        yield
    finally:
        if acquired:
            try:
                _unlock(fd)
            finally:
                os.close(fd)
        else:
            os.close(fd)


def fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_unlink(path: Path, *, missing_ok: bool = False) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise
        return
    fsync_directory(path.parent)


def append_bytes_durable(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Append one complete payload and durably persist file and directory state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, mode)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        else:  # pragma: no cover - Windows
            os.chmod(path, mode)
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    if not existed:
        fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    """Durably replace ``path`` with UTF-8 text in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644

    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    fd_is_open = True
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        else:  # pragma: no cover - Windows
            os.chmod(temp_path, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fd_is_open = False
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        fsync_directory(path.parent)
    except BaseException:
        if fd_is_open:
            os.close(fd)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int | None = 2,
    mode: int | None = None,
) -> None:
    kwargs: dict[str, Any] = {"allow_nan": False, "ensure_ascii": False}
    if indent is None:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = indent
    atomic_write_text(path, json.dumps(data, **kwargs), mode=mode)


def atomic_write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    mode: int | None = None,
) -> None:
    text = "".join(f"{json.dumps(row, ensure_ascii=False, allow_nan=False)}\n" for row in rows)
    atomic_write_text(path, text, mode=mode)
