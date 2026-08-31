"""Track-specific prompt templates. Bump the version string on any change so
cached predictions from an old prompt are not reused."""

LINE_V1 = (
    "You are a transcription tool for handwritten Arabic.\n"
    "Transcribe the Arabic text in this image exactly as written, on a single line.\n"
    "Preserve any diacritics (harakat) that are actually present.\n"
    "Output only the transcription. No translation, no commentary, no quotation marks."
)

# v2: the image is ONE line; forbid the markup / repetition failure modes seen
# from structure-first OCR models (Qari v0.3).
LINE_V2 = (
    "This image contains a single line of handwritten Arabic.\n"
    "Transcribe exactly that one line, as plain text.\n"
    "Preserve harakat that are actually written; do not invent any.\n"
    "Do NOT add HTML tags, Markdown, bullet points, or formatting.\n"
    "Do NOT repeat words or phrases. If a word is illegible, give your best single guess.\n"
    "Output only the transcription — nothing before or after it."
)

PAGE_V1 = (
    "You are a transcription tool for handwritten Arabic manuscripts.\n"
    "Transcribe every line of Arabic text in this page image.\n"
    "Reading order: top to bottom; within a line, right to left.\n"
    "Put each text line on its own line. Preserve diacritics that are present.\n"
    "Ignore page numbers, headers, marginalia and graphics.\n"
    "Output only the transcription. No translation, no commentary."
)

_PROMPTS = {
    "line-v1": LINE_V1,
    "line-v2": LINE_V2,
    "page-v1": PAGE_V1,
}


def get_prompt(version: str) -> str:
    try:
        return _PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"unknown prompt_version {version!r}; known: {sorted(_PROMPTS)}"
        ) from None
