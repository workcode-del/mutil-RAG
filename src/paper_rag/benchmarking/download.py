from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from pathlib import Path


logger = logging.getLogger(__name__)


def download_file(url: str, target: str | Path, *, force: bool = False) -> Path:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        if _valid_download(destination):
            logger.info("Using cached download: %s", destination)
            return destination
        logger.warning("Invalid cached download; replacing: %s", destination)
        force = True

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() and not force else 0
    if force and partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "paper-rag-benchmark/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    logger.info("Downloading: %s -> %s", url, destination)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if offset and not append:
            offset = 0
        with partial.open("ab" if append else "wb") as stream:
            downloaded = offset
            next_report = downloaded + 256 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    logger.info(
                        "Download progress: %s %.0f MiB",
                        destination.name,
                        downloaded / 1024**2,
                    )
                    next_report += 256 * 1024 * 1024
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {url}")
    os.replace(partial, destination)
    if not _valid_download(destination):
        destination.unlink()
        raise RuntimeError(
            f"Downloaded content is not a valid {destination.suffix} file: {url}. "
            "The server may have returned an HTML error or access page."
        )
    logger.info("Download ready: %s %.1f MiB", destination, destination.stat().st_size / 1024**2)
    return destination


def extract_zip(archive: str | Path, destination: str | Path, *, force: bool = False) -> Path:
    source = Path(archive)
    if not zipfile.is_zipfile(source):
        raise zipfile.BadZipFile(f"Not a valid ZIP archive: {source}")
    root = Path(destination)
    marker = root / f".{source.name}.extracted"
    if marker.exists() and not force:
        logger.info("Using extracted dataset: %s", root)
        return root
    logger.info("Extracting archive: %s -> %s", source, root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    with zipfile.ZipFile(source) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if os.path.commonpath([resolved_root, target]) != str(resolved_root):
                raise ValueError(f"Unsafe zip member: {member.filename}")
        bundle.extractall(root)
    marker.touch()
    logger.info("Archive extracted: %s", root)
    return root


def _valid_download(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return zipfile.is_zipfile(path)
    if suffix == ".pdf":
        with path.open("rb") as stream:
            return stream.read(5) == b"%PDF-"
    return True
