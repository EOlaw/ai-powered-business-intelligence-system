"""
InsightSerenity AI Engine — File I/O Utilities
===============================================
Async and sync helpers for reading/writing every file format used across
the ai-engine: plain text, JSONL (the primary corpus format), JSON config
files, and binary pickle/numpy files.

JSONL (JSON Lines) is the standard corpus interchange format throughout this
platform: one JSON object per line, streamable, appendable, and human-readable.

Usage:
    from src.utils.file_io import read_jsonl, write_jsonl, atomic_write

    records = await read_jsonl("storage/datasets/corpus.jsonl")
    await write_jsonl("storage/datasets/cleaned.jsonl", records)
"""

import asyncio
import gzip
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Generator, Iterable, Iterator, List, Optional, Union

import aiofiles

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

PathLike = Union[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: PathLike) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.
    Returns the resolved Path object.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_size_mb(path: PathLike) -> float:
    """Return the size of a file in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)


def count_lines(path: PathLike) -> int:
    """
    Efficiently count lines in a file without loading it into memory.
    Works on plain text and .gz files.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "rb") as f:                    # type: ignore[call-overload]
        for _ in f:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Atomic write — prevents partial file writes corrupting data on crash
# ─────────────────────────────────────────────────────────────────────────────

def atomic_write(path: PathLike, content: Union[str, bytes], mode: str = "w") -> None:
    """
    Write content to `path` atomically using a temp file + rename.

    This guarantees that readers never see a partially-written file.
    If the process crashes mid-write, the original file is untouched.

    Args:
        path:    Destination file path.
        content: String (mode="w") or bytes (mode="wb") to write.
        mode:    File open mode — "w" for text, "wb" for binary.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling temp file in the same directory (same filesystem)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(tmp_fd, mode) as f:
            f.write(content)
        # Atomic rename — on POSIX this is guaranteed to be atomic
        shutil.move(tmp_path, str(destination))
        logger.debug("Atomic write completed", path=str(destination))
    except Exception:
        # Clean up the temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_json(path: PathLike) -> Any:
    """Synchronously read a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """Atomically write `data` as a pretty-printed JSON file."""
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write(path, content, mode="w")


async def async_read_json(path: PathLike) -> Any:
    """Asynchronously read and parse a JSON file."""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    return json.loads(content)


async def async_write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """Asynchronously write a JSON file (non-atomic — use for non-critical data)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=indent, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# JSONL — the primary corpus interchange format
# ─────────────────────────────────────────────────────────────────────────────

def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """
    Lazily iterate over a JSONL file one record at a time.
    Memory-efficient for large corpora — never loads the full file.

    Supports plain .jsonl and compressed .jsonl.gz files.

    Yields:
        Parsed dict for each non-empty line.

    Example:
        for record in iter_jsonl("corpus.jsonl"):
            process(record["text"])
    """
    path_str = str(path)
    opener   = gzip.open if path_str.endswith(".gz") else open

    with opener(path_str, "rt", encoding="utf-8") as f:    # type: ignore[call-overload]
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping malformed JSONL line",
                    path=path_str,
                    line_num=line_num,
                    error=str(e),
                )


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """
    Load an entire JSONL file into a list.
    Use `iter_jsonl` instead for large files to avoid OOM.
    """
    return list(iter_jsonl(path))


def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> int:
    """
    Write an iterable of dicts to a JSONL file (plain text, UTF-8).
    Each record is serialised as a single line.

    Returns:
        Number of records written.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    logger.debug("JSONL written", path=str(path), records=count)
    return count


def append_jsonl(path: PathLike, record: Dict[str, Any]) -> None:
    """
    Append a single record to a JSONL file.
    Creates the file if it does not exist.
    Thread-safe via OS-level append semantics (not process-safe on Windows).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def async_iter_jsonl(path: PathLike) -> AsyncIterator[Dict[str, Any]]:
    """
    Asynchronously iterate over a JSONL file line by line.
    Use this inside async coroutines (e.g., the web crawler pipeline).
    """
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


async def async_write_jsonl(
    path: PathLike,
    records: Iterable[Dict[str, Any]],
) -> int:
    """Asynchronously write records to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        for record in records:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Plain text helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_text(path: PathLike, encoding: str = "utf-8") -> str:
    """Read the entire content of a text file as a string."""
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path: PathLike, content: str) -> None:
    """Atomically write a string to a text file."""
    atomic_write(path, content, mode="w")


def read_lines(path: PathLike, strip: bool = True) -> List[str]:
    """
    Read a text file and return a list of non-empty lines.

    Args:
        path:  Path to the text file.
        strip: If True, strip leading/trailing whitespace from each line.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if strip:
        lines = [l.strip() for l in lines if l.strip()]
    return lines


async def async_read_text(path: PathLike) -> str:
    """Asynchronously read the full content of a text file."""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Binary / pickle helpers (for model checkpoints and vocab files)
# ─────────────────────────────────────────────────────────────────────────────

def save_pickle(path: PathLike, obj: Any) -> None:
    """Serialize an object to a pickle file using protocol 4 (Python 3.8+)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(path: PathLike) -> Any:
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Checksum helpers — verify file integrity before loading
# ─────────────────────────────────────────────────────────────────────────────

def md5_of_file(path: PathLike, chunk_size: int = 8192) -> str:
    """
    Compute the MD5 hex digest of a file without loading it all into memory.
    Used to verify checkpoint integrity and detect exact-duplicate documents.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_of_string(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def md5_of_string(text: str) -> str:
    """Return the MD5 hex digest of a UTF-8 string (fast, for deduplication)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Chunked file splitting — useful for distributing corpus across workers
# ─────────────────────────────────────────────────────────────────────────────

def split_jsonl(
    input_path: PathLike,
    output_dir: PathLike,
    chunk_size: int = 100_000,
) -> List[Path]:
    """
    Split a large JSONL file into smaller chunks of `chunk_size` lines each.

    This is used to distribute preprocessing work across multiple worker
    processes without holding the full corpus in memory.

    Args:
        input_path:  Source JSONL file.
        output_dir:  Directory where chunk files will be written.
        chunk_size:  Number of records per chunk.

    Returns:
        List of paths to the created chunk files.
    """
    output_dir = ensure_dir(output_dir)
    chunk_paths: List[Path] = []
    chunk_num   = 0
    current_chunk: List[str] = []

    for line in iter_jsonl(input_path):
        current_chunk.append(json.dumps(line, ensure_ascii=False))

        if len(current_chunk) >= chunk_size:
            chunk_path = output_dir / f"chunk_{chunk_num:05d}.jsonl"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write("\n".join(current_chunk) + "\n")
            chunk_paths.append(chunk_path)
            logger.debug("Chunk written", chunk=chunk_num, records=len(current_chunk))
            current_chunk = []
            chunk_num += 1

    # Write the final (possibly smaller) chunk
    if current_chunk:
        chunk_path = output_dir / f"chunk_{chunk_num:05d}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write("\n".join(current_chunk) + "\n")
        chunk_paths.append(chunk_path)

    logger.info(
        "JSONL split completed",
        input=str(input_path),
        chunks=len(chunk_paths),
        chunk_size=chunk_size,
    )
    return chunk_paths
