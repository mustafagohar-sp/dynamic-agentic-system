from uuid import uuid4

from app.rag.context import assemble_context
from app.rag.retrieval import RetrievalResult


def make_result(
    filename: str,
    chunk_index: int,
    content: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=chunk_index,
        content=content,
        filename=filename,
    )


def test_assemble_context_formats_retrieval_results():
    results = [
        make_result(
            "financial_report.pdf",
            2,
            "Northbridge FC generated £204.0 million in revenue.",
        ),
        make_result(
            "commercial_report.pdf",
            4,
            "Commercial income increased during 2024/25.",
        ),
    ]

    context = assemble_context(results)

    assert "[Source 1]" in context.text
    assert "[Source 2]" in context.text

    assert "Document: financial_report.pdf" in context.text
    assert "Document: commercial_report.pdf" in context.text

    assert "Chunk: 2" in context.text
    assert "Chunk: 4" in context.text

    assert (
        "Northbridge FC generated £204.0 million in revenue."
        in context.text
    )

    assert len(context.sources) == 2
    assert context.sources[0].source_number == 1
    assert context.sources[1].source_number == 2


def test_assemble_context_handles_empty_results():
    context = assemble_context([])

    assert context.text == ""
    assert context.sources == []


def test_assemble_context_respects_character_limit():
    results = [
        make_result(
            "financial_report.pdf",
            0,
            "A" * 100,
        ),
        make_result(
            "commercial_report.pdf",
            1,
            "B" * 100,
        ),
    ]

    context = assemble_context(
        results,
        max_characters=150,
    )

    assert len(context.text) <= 150
    assert len(context.sources) == 1
    assert "A" in context.text
    assert "B" not in context.text