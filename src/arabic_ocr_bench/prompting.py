"""Track-specific prompt templates. Bump the version string on any change so
cached predictions from an old prompt are not reused."""

LINE_V1 = (
    "You are a transcription tool for handwritten Arabic.\n"
    "Transcribe the Arabic text in this image exactly as written, on a single line.\n"
    "Preserve any diacritics (harakat) that are actually present.\n"
    "Output only the transcription. No translation, no commentary, no quotation marks."
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
    "page-v1": PAGE_V1,
}


def get_prompt(version: str) -> str:
    try:
        return _PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"unknown prompt_version {version!r}; known: {sorted(_PROMPTS)}"
        ) from None
