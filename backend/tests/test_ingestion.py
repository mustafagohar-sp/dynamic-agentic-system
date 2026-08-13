from pathlib import Path

import pytest

from app.rag.ingestion import (
    DocumentIngestionError,
    calculate_checksum,
    extract_text,
    validate_file,
)


def test_extract_text_from_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Northbridge FC test document.", encoding="utf-8")

    text = extract_text(file_path)

    assert text == "Northbridge FC test document."


def test_extract_text_from_pdf():
    pdf_path = (
        Path(__file__).parent.parent
        / "test_data"
        / "northbridge_fc"
        / "financial"
        / "Northbridge_FC_Annual_Financial_Report_2024_25.pdf"
    )

    text = extract_text(pdf_path)

    assert text
    assert "NORTHBRIDGE FC" in text
    assert "Annual Financial Report 2024/25" in text


def test_validate_file_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(DocumentIngestionError, match="Unsupported file type"):
        validate_file(file_path)


def test_extract_text_rejects_empty_text_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentIngestionError, match="contains no extractable text"):
        extract_text(file_path)


def test_calculate_checksum(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Northbridge FC", encoding="utf-8")

    checksum = calculate_checksum(file_path)

    assert len(checksum) == 64
    assert all(character in "0123456789abcdef" for character in checksum)