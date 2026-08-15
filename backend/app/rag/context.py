from dataclasses import dataclass

from app.rag.retrieval import RetrievalResult


@dataclass(frozen=True)
class ContextSource:
    source_number: int
    filename: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class AssembledContext:
    text: str
    sources: list[ContextSource]


def assemble_context(
    results: list[RetrievalResult],
    max_characters: int = 12000,
) -> AssembledContext:
    sources = []
    sections = []
    current_length = 0

    for result in results:
        source_number = len(sources) + 1

        header = (
            f"[Source {source_number}]\n"
            f"Document: {result.filename}\n"
            f"Chunk: {result.chunk_index}\n\n"
        )

        separator = "\n\n" if sections else ""

        available_length = (
            max_characters
            - current_length
            - len(separator)
            - len(header)
            - 1
        )

        if available_length <= 0:
            break

        content = result.content[:available_length]

        section = (
            f"{header}"
            f"{content}\n"
        )

        sections.append(
            f"{separator}{section}"
        )

        sources.append(
            ContextSource(
                source_number=source_number,
                filename=result.filename,
                chunk_index=result.chunk_index,
                content=content,
            )
        )

        current_length += len(separator) + len(section)

        if len(content) < len(result.content):
            break

    return AssembledContext(
        text="".join(sections),
        sources=sources,
    )