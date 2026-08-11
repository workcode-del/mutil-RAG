from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, target: str | Path, *, force: bool = False) -> Path:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"Using cached download: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() and not force else 0
    if force and partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "paper-rag-benchmark/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    print(f"Downloading {url} -> {destination}")
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
                    print(f"  {destination.name}: {downloaded / 1024**2:.0f} MiB")
                    next_report += 256 * 1024 * 1024
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {url}")
    os.replace(partial, destination)
    print(f"Downloaded {destination.stat().st_size / 1024**2:.1f} MiB: {destination}")
    return destination


def extract_zip(archive: str | Path, destination: str | Path, *, force: bool = False) -> Path:
    source = Path(archive)
    root = Path(destination)
    marker = root / f".{source.name}.extracted"
    if marker.exists() and not force:
        return root
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    with zipfile.ZipFile(source) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if os.path.commonpath([resolved_root, target]) != str(resolved_root):
                raise ValueError(f"Unsafe zip member: {member.filename}")
        bundle.extractall(root)
    marker.touch()
    return root
