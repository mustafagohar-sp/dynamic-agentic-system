from pathlib import Path
import hashlib

from sqlalchemy.orm import Session
from pypdf import PdfReader

from app.models.document import Document
from app.rag.chunking import create_document_chunks


class DocumentIngestionError(Exception):
    """Raised when document ingestion fails."""


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def validate_file(file_path: str | Path) -> Path:
    path = Path(file_path)

    if not path.exists():
        raise DocumentIngestionError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise DocumentIngestionError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentIngestionError(
            f"Unsupported file type: {path.suffix}"
        )

    return path


def extract_text(file_path: str | Path) -> str:
    path = validate_file(file_path)

    try:
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")

        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))

            pages = []

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    pages.append(page_text)

            text = "\n\n".join(pages)

        else:
            raise DocumentIngestionError(
                f"Unsupported file type: {path.suffix}"
            )

    except UnicodeDecodeError as exc:
        raise DocumentIngestionError(
            f"Could not decode file as UTF-8: {path}"
        ) from exc

    except OSError as exc:
        raise DocumentIngestionError(
            f"Could not read file: {path}"
        ) from exc

    except Exception as exc:
        if isinstance(exc, DocumentIngestionError):
            raise

        raise DocumentIngestionError(
            f"Could not extract text from file: {path}"
        ) from exc

    text = text.strip()

    if not text:
        raise DocumentIngestionError(
            f"Document contains no extractable text: {path}"
        )

    return text


def calculate_checksum(file_path: str | Path) -> str:
    path = validate_file(file_path)

    try:
        file_bytes = path.read_bytes()

    except OSError as exc:
        raise DocumentIngestionError(
            f"Could not read file for checksum: {path}"
        ) from exc

    return hashlib.sha256(file_bytes).hexdigest()


def ingest_document_with_chunks(
    db: Session,
    file_path: str | Path,
    kb_version_id,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Document:

    path = validate_file(file_path)

    text = extract_text(path)
    checksum = calculate_checksum(path)

    document = Document(
        kb_version_id=kb_version_id,
        filename=path.name,
        content_type=(
            "application/pdf"
            if path.suffix.lower() == ".pdf"
            else "text/plain"
        ),
        storage_path=str(path),
        checksum=checksum,
    )

    db.add(document)
    db.flush()

    create_document_chunks(
        db=db,
        document_id=document.id,
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    db.commit()
    db.refresh(document)

    return document