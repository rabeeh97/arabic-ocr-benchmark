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

# --- Prompt-sensitivity study (2026-09) ------------------------------------
# One-variable-at-a-time variants of LINE_V2, each isolating a single axis so
# results are directly attributable. See docs/experiments/prompt-study.md for
# the design and findings. All variants keep LINE_V2's other constraints
# (plain text, no formatting, no repetition, best-guess on illegible words)
# unless the axis under test is precisely that constraint (line-v2-no-diac).

# Modality axis: LINE_V2 always claims "handwritten", including on datasets
# that are actually printed (kitab_hindawi, kitab_historicalbooks) — an
# existing confound in every prior benchmark run. These two variants plus the
# LINE_V2 baseline give the full correct/wrong/neutral modality-claim grid
# without needing a claim string per dataset: LINE_V2 is "correct" on
# handwritten sets and "wrong" on printed ones; LINE_V2_CLAIMS_PRINTED is the
# mirror (correct on printed, wrong on handwritten); LINE_V2_NO_MODALITY drops
# the claim entirely as a neutral control.
LINE_V2_NO_MODALITY = (
    "This image contains a single line of Arabic text.\n"
    "Transcribe exactly that one line, as plain text.\n"
    "Preserve harakat that are actually written; do not invent any.\n"
    "Do NOT add HTML tags, Markdown, bullet points, or formatting.\n"
    "Do NOT repeat words or phrases. If a word is illegible, give your best single guess.\n"
    "Output only the transcription — nothing before or after it."
)

LINE_V2_CLAIMS_PRINTED = (
    "This image contains a single line of printed Arabic text.\n"
    "Transcribe exactly that one line, as plain text.\n"
    "Preserve harakat that are actually written; do not invent any.\n"
    "Do NOT add HTML tags, Markdown, bullet points, or formatting.\n"
    "Do NOT repeat words or phrases. If a word is illegible, give your best single guess.\n"
    "Output only the transcription — nothing before or after it."
)

# Language-of-instructions axis: identical content to LINE_V2, in Arabic.
LINE_V2_ARABIC = (
    "هذه الصورة تحتوي على سطر واحد من نص عربي مكتوب بخط اليد.\n"
    "انسخ هذا السطر تمامًا كما هو، كنص عادي دون أي تنسيق.\n"
    "حافظ على الحركات (التشكيل) الموجودة فعليًا في النص، ولا تضف أي حركة غير موجودة.\n"
    "لا تضف أي وسوم HTML أو تنسيق Markdown أو نقاط تعداد.\n"
    "لا تكرر أي كلمة أو عبارة. إذا كانت كلمة غير واضحة، اكتب أفضل تخمين واحد لها.\n"
    "أخرج النص المنسوخ فقط، دون أي إضافة قبله أو بعده."
)

# Persona-framing axis: LINE_V2 prefixed with an expert-persona instruction.
LINE_V2_PERSONA = (
    "You are an expert paleographer and philologist specializing in Arabic "
    "manuscripts and historical Arabic scripts.\n"
    "This image contains a single line of handwritten Arabic.\n"
    "Transcribe exactly that one line, as plain text.\n"
    "Preserve harakat that are actually written; do not invent any.\n"
    "Do NOT add HTML tags, Markdown, bullet points, or formatting.\n"
    "Do NOT repeat words or phrases. If a word is illegible, give your best single guess.\n"
    "Output only the transcription — nothing before or after it."
)

# Diacritics axis: instead of "preserve what's there, don't invent", demand
# skeleton letters only. Cross-check against the diac_insensitive CER profile.
LINE_V2_NO_DIAC = (
    "This image contains a single line of handwritten Arabic.\n"
    "Transcribe exactly that one line, as plain text.\n"
    "Do NOT include any diacritics (harakat) — output the bare letters only, "
    "even if the original has visible harakat.\n"
    "Do NOT add HTML tags, Markdown, bullet points, or formatting.\n"
    "Do NOT repeat words or phrases. If a word is illegible, give your best single guess.\n"
    "Output only the transcription — nothing before or after it."
)

# Granularity axis: kitab_hindawi / kitab_historicalbooks are paragraph/page
# scale (see docs/experiments/findings-and-notes.md) but every prior run
# scored them with a single-LINE prompt. PAGE_V1 also wrongly claims
# "handwritten" for these printed book subsets — PAGE_V2 fixes both at once.
PAGE_V2 = (
    "This image contains one or more lines of printed Arabic text — possibly "
    "a full paragraph or page, not just a single line.\n"
    "Transcribe every line of text you see, in reading order (top to bottom; "
    "within a line, right to left).\n"
    "Put each text line on its own output line. Preserve diacritics that are "
    "actually present; do not invent any.\n"
    "Ignore page numbers, headers, marginalia and graphics.\n"
    "Output only the transcription. No translation, no commentary."
)

_PROMPTS = {
    "line-v1": LINE_V1,
    "line-v2": LINE_V2,
    "page-v1": PAGE_V1,
    "line-v2-no-modality": LINE_V2_NO_MODALITY,
    "line-v2-claims-printed": LINE_V2_CLAIMS_PRINTED,
    "line-v2-arabic": LINE_V2_ARABIC,
    "line-v2-persona": LINE_V2_PERSONA,
    "line-v2-no-diac": LINE_V2_NO_DIAC,
    "page-v2": PAGE_V2,
}


def get_prompt(version: str) -> str:
    try:
        return _PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"unknown prompt_version {version!r}; known: {sorted(_PROMPTS)}"
        ) from None
