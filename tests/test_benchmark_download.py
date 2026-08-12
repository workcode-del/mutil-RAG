import zipfile

from paper_rag.benchmarking.download import _valid_download, extract_zip


def test_zip_validation_rejects_html_cache(tmp_path) -> None:
    archive = tmp_path / "dataset.zip"
    archive.write_text("<html>access denied</html>", encoding="utf-8")

    assert not _valid_download(archive)


def test_zip_validation_accepts_and_extracts_archive(tmp_path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("qa.jsonl", "{}\n")

    assert _valid_download(archive)
    output = extract_zip(archive, tmp_path / "output")
    assert (output / "qa.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_pdf_validation_checks_file_signature(tmp_path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"<html>rate limited</html>")
    assert not _valid_download(pdf)

    pdf.write_bytes(b"%PDF-1.7\n")
    assert _valid_download(pdf)
