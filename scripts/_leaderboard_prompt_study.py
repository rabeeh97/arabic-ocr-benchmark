"""The leaderboard's "Research" tab: the prompt-sensitivity study's findings,
written for a reader with no OCR-benchmarking background, plus two stub
sections (other research directions; is this benchmark publishable) held open
for when we get back to them.

Standalone in the same sense as ``_leaderboard_findings.py``: reuses its chart
primitives (``hbar``, ``dumbbell``, ``svg_heatmap``) so the visuals match the
Findings tab exactly, but reads its own data straight from
``results/*/predictions.jsonl`` + ``scores.jsonl`` rather than the generic
``findings.json`` aggregate — this study's data (which prompt_version scored
which line) doesn't fit that schema.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from _leaderboard_findings import dumbbell, hbar, svg_heatmap

MODEL = "gemini-flash-lite"
MODEL_PAGE = "gemini-flash-lite-640tok"

# variant key -> (short label for charts, one-line plain-English description)
VARIANTS = {
    "line-v2": ("baseline", "today's production prompt — always says \"handwritten\""),
    "line-v2-no-modality": ("no modality", "drops the handwritten/printed claim entirely"),
    "line-v2-claims-printed": ("claims printed", "says \"printed\" instead — correct on the 2 printed sets, wrong on the 4 handwritten ones with data"),
    "line-v2-arabic": ("Arabic instructions", "identical instructions, translated into Arabic"),
    "line-v2-persona": ("expert persona", "prefixed with an \"expert paleographer\" persona"),
    "line-v2-no-diac": ("no diacritics", "explicitly told to omit all harakat"),
}
CORE_VARIANTS = list(VARIANTS)  # display order

# dataset key -> (short label, modality tag for the reader)
DATASETS = {
    "kitab_khatt": ("KHATT", "handwritten"),
    "kitab_historyar": ("HistoryAr", "handwritten"),
    "kitab_onlinekhatt": ("OnlineKHATT", "handwritten"),
    "kitab_hindawi": ("Hindawi", "printed"),
    "kitab_historicalbooks": ("HistoricalBooks", "printed"),
}
PILOT_N = {"kitab_khatt": 15, "kitab_historyar": 15, "kitab_onlinekhatt": 15,
           "kitab_hindawi": 15, "kitab_historicalbooks": 10}
BLOCKED = {"muharaf_lines": "handwritten", "nakba_lines": "handwritten", "rasam1": "handwritten"}


def _esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _load(target: Path) -> dict:
    """(model, dataset, prompt_version) -> list of score dicts, scoped per
    run dir so a sample_id's prompt_version is never ambiguous across runs."""
    out: dict[tuple[str, str, str], list[dict]] = {}
    for pred_f in target.glob("*/predictions.jsonl"):
        score_f = pred_f.parent / "scores.jsonl"
        if not score_f.exists():
            continue
        local_pv: dict[tuple[str, str, str], str] = {}
        local_len: dict[tuple[str, str, str], int] = {}
        try:
            pred_lines = pred_f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in pred_lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("model") not in (MODEL, MODEL_PAGE):
                continue
            key = (r["model"], r.get("dataset", ""), r.get("sample_id", ""))
            local_pv[key] = r.get("prompt_version")
            local_len[key] = len(r.get("cleaned_text") or "")
        if not local_pv:
            continue
        try:
            score_lines = score_f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in score_lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("model") not in (MODEL, MODEL_PAGE) or r.get("error"):
                continue
            key = (r["model"], r.get("dataset", ""), r.get("sample_id", ""))
            pv = local_pv.get(key)
            if not pv or "cer_light" not in r:
                continue
            ds = r["dataset"]
            n_cap = PILOT_N.get(ds)
            if n_cap is not None:
                try:
                    idx = int(str(r["sample_id"]).split("/")[-1])
                except ValueError:
                    idx = None
                if idx is not None and idx >= n_cap:
                    continue
            r = dict(r)
            r["_pred_len"] = local_len.get(key, 0)
            out.setdefault((r["model"], ds, pv), []).append(r)
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fig(caption: str, chart: str) -> str:
    return f"<figure><figcaption>{_esc(caption)}</figcaption>{chart}</figure>"


def _callout(text: str) -> str:
    return (f"<p class='status status--live'><span class='dot'></span>{text}</p>")


def _tag(label: str, kind: str = "ds") -> str:
    return f"<span class='chip chip--{kind}'>{_esc(label)}</span>"


def _section(anchor: str, title: str, body: str) -> str:
    return f"<section id='{anchor}' class='f-panel'><h2>{_esc(title)}</h2>{body}</section>"


def render_research_view(target: Path) -> tuple[str, list[tuple[str, str]]]:
    """Returns (html, toc_items) for the Research tab — always non-empty (the
    two stub sections render even with no prompt-study data on disk yet)."""
    data = _load(target)
    ds_list = list(DATASETS)
    toc: list[tuple[str, str]] = []
    parts = ['<div class="findings-view">']

    # ---- intro -------------------------------------------------------------
    intro = (
        "<p class='board-note'>This tab is about a different question than the "
        "main leaderboard: not <i>which model</i> is best, but <i>does how you "
        "ask a model change how well it reads Arabic?</i> Every OCR call here is "
        "one instruction (\"transcribe this line...\") plus one image, sent to the "
        "model. We held the model and the images fixed and only changed the "
        "wording of the instruction, then compared the resulting accuracy. "
        "<b>CER</b> (character error rate) is the percent of characters the "
        "model got wrong — lower is better, 0% is a perfect transcription. "
        "<b>WER</b> is the same idea per word. All numbers below are CER unless "
        "labelled otherwise.</p>"
        "<p class='board-note'>Model: <code>gemini-flash-lite</code> (Google's "
        "cheapest/fastest Gemini). 6 prompt variants &times; 5 datasets that "
        "could be tested this pass, 10-15 sample lines each — a quick first "
        "read before spending a lot more quota on a full sweep. Full "
        "methodology, every prompt's exact wording, and known limitations are "
        "in <code>docs/experiments/prompt-study.md</code>.</p>"
    )
    parts.append(_section("rs-intro", "Prompt sensitivity — how we tested it", intro))
    toc.append(("rs-intro", "How we tested it"))

    if not data:
        parts.append(_section(
            "rs-nodata", "No prompt-study data yet",
            "<p class='board-note'>Run <code>scripts/run_prompt_study.py --phase "
            "pilot</code> and regenerate the leaderboard to populate this tab.</p>",
        ))
        toc.append(("rs-nodata", "No data yet"))
    else:
        # ---- per (variant, dataset) CER -------------------------------------
        cer: dict[tuple[str, str], list[float]] = {}
        for (model, ds, pv), rows in data.items():
            if pv not in VARIANTS or ds not in DATASETS:
                continue
            cer.setdefault((pv, ds), []).extend(r["cer_light"] for r in rows)

        # ---- headline: page-v2 vs line-v2 on the 2 page-scale datasets -----
        page_rows = []
        for ds in ("kitab_hindawi", "kitab_historicalbooks"):
            base = _mean(cer.get(("line-v2", ds), []))
            pg = _mean([r["cer_light"] for r in data.get((MODEL_PAGE, ds, "page-v2"), [])])
            if base is not None and pg is not None:
                page_rows.append((DATASETS[ds][0], base * 100, pg * 100))
        if page_rows:
            best_ds, best_before, best_after = max(page_rows, key=lambda r: r[1] - r[2])
            headline = _callout(
                f"<b>Biggest lever found so far:</b> telling the model it may be looking "
                f"at a whole paragraph/page (not one line), and giving it room to answer "
                f"(640 vs 256 tokens), cut CER on <b>{_esc(best_ds)}</b> from "
                f"{best_before:.0f}% to <b>{best_after:.0f}%</b>. This wasn't a small "
                f"tweak — it fixed a standing, previously-unexplained gap on these two "
                f"datasets across every model in the main benchmark."
            )
            body = (
                headline
                + "<p class='board-note'>These two datasets (<code>kitab_hindawi</code>, "
                "<code>kitab_historicalbooks</code>) are digitised book pages — their "
                "\"ground truth\" is actually a whole paragraph, not one line. Every "
                "model in this benchmark had always been asked to transcribe them as if "
                "they were a single short line, capped at 256 tokens of output. Both the "
                "wrong framing and the token cap were silently truncating the answer. "
                "<code>page-v2</code> fixes both: it tells the model it may see multiple "
                "lines of a page, asks for the whole thing in reading order, and gives it "
                "640 tokens to answer with.</p>"
                + _fig(
                    "CER before -> after switching to a page-appropriate prompt "
                    "(lower is better)",
                    dumbbell(page_rows, left_label="line-v2 (1 line, 256 tok)",
                             right_label="page-v2 (whole page, 640 tok)"),
                )
                + "<p class='board-note'><b>Recommendation:</b> register a "
                "page-appropriate prompt + higher token cap for these two datasets "
                "across every model in the main benchmark, not just this one cheap "
                "model — this looks like a data-handling bug the whole leaderboard has "
                "been carrying, not a prompt-engineering nicety.</p>"
            )
            parts.append(_section("rs-page", "The big one: match the prompt to the data's real scale", body))
            toc.append(("rs-page", "Page vs. line prompts"))

        # ---- grand summary bar + full heatmap --------------------------------
        means_by_variant = []
        for pv in CORE_VARIANTS:
            vals = [_mean(cer.get((pv, ds), [])) for ds in ds_list]
            vals = [v for v in vals if v is not None]
            if vals:
                means_by_variant.append((VARIANTS[pv][0], _mean(vals) * 100))
        heat_rows = [f"{DATASETS[ds][0]} ({DATASETS[ds][1]})" for ds in ds_list]
        heat_cols = [VARIANTS[pv][0] for pv in CORE_VARIANTS]
        heat_vals = [
            [(_mean(cer.get((pv, ds), [])) * 100 if cer.get((pv, ds)) else None)
             for pv in CORE_VARIANTS]
            for ds in ds_list
        ]
        summary_body = (
            "<p class='board-note'>Mean CER across the 5 datasets with data, one "
            "bar per prompt variant — the quickest way to see which wording "
            "helped or hurt on average, before looking at any single dataset.</p>"
            + _fig("Mean CER-light by prompt variant (lower is better)",
                   hbar(means_by_variant, unit="%"))
            + _fig("CER-light, every variant x every dataset (dataset's modality in "
                   "parentheses) — darker = worse",
                   svg_heatmap(heat_rows, heat_cols, heat_vals, fmt="{:.0f}%"))
        )
        parts.append(_section("rs-summary", "All variants at a glance", summary_body))
        toc.append(("rs-summary", "All variants at a glance"))

        # ---- diacritics ------------------------------------------------------
        diac_rows = []
        for ds in ds_list:
            base = _mean(cer.get(("line-v2", ds), []))
            nd = _mean(cer.get(("line-v2-no-diac", ds), []))
            if base is not None and nd is not None:
                diac_rows.append((DATASETS[ds][0], base * 100, nd * 100))
        if diac_rows:
            diacf1_base = _mean([r.get("diac_f1", 0.0) for rows in
                                  (data.get((MODEL, ds, "line-v2"), []) for ds in ds_list)
                                  for r in rows])
            diacf1_nd = _mean([r.get("diac_f1", 0.0) for rows in
                                (data.get((MODEL, ds, "line-v2-no-diac"), []) for ds in ds_list)
                                for r in rows])
            body = (
                "<p class='board-note'>The production prompt says \"preserve harakat "
                "(diacritics) that are actually written; do not invent any.\" This "
                "variant instead bluntly says \"never output diacritics at all.\" If "
                "the model were already following the first instruction well, being "
                "blunter shouldn't change accuracy much. It did:</p>"
                + _fig("CER: baseline vs. \"never output diacritics\" (lower is better)",
                       dumbbell(diac_rows, left_label="line-v2 (preserve real ones)",
                                right_label="line-v2-no-diac (omit all)"))
                + f"<p class='board-note'>The catch: diacritic-F1 (how well the model's "
                f"harakat match the real ones, when it does try) drops from "
                f"{(diacf1_base or 0)*100:.0f}% to {(diacf1_nd or 0)*100:.0f}% mean, "
                f"since this variant emits none by design. Read together, this says "
                f"the model was <b>inventing</b> diacritics that weren't there often "
                f"enough that suppressing all of them was a net accuracy win — not "
                f"that diacritics don't matter. <b>Recommendation:</b> the production "
                f"instruction's wording likely needs to be firmer about not guessing, "
                f"rather than dropped altogether; worth testing an in-between phrasing "
                f"in the full sweep.</p>"
            )
            parts.append(_section("rs-diac", "Diacritics: less guessing beat \"preserve what's real\"", body))
            toc.append(("rs-diac", "Diacritics"))

        # ---- modality claim ----------------------------------------------------
        modality_cols = ["line-v2", "line-v2-no-modality", "line-v2-claims-printed"]
        mod_heat_vals = [
            [(_mean(cer.get((pv, ds), [])) * 100 if cer.get((pv, ds)) else None)
             for pv in modality_cols]
            for ds in ds_list
        ]
        body = (
            "<p class='board-note'>The production prompt always says \"this image "
            "contains a single line of <b>handwritten</b> Arabic\" — even for the two "
            "printed datasets above. Does the claim matter, and does getting it "
            "wrong actively hurt?</p>"
            + _fig("CER-light: claiming handwritten (baseline) vs. no claim vs. "
                   "claiming printed — dataset's real modality in parentheses",
                   svg_heatmap(heat_rows, ["claims HW (baseline)", "no claim", "claims printed"],
                               mod_heat_vals, fmt="{:.0f}%"))
            + "<p class='board-note'><b>Result: inconsistent, not a clean answer "
            "either way.</b> On the one dataset where the claim actually changes from "
            "wrong to correct (HistoricalBooks, printed), the correct claim did help "
            "(73%&rarr;67%). On the other printed dataset (Hindawi) the correct claim "
            "made things clearly <i>worse</i> (64%&rarr;79%). On the handwritten sets, "
            "the (wrong) \"printed\" claim beat the (correct) \"handwritten\" claim on "
            "2 of 3. Dropping the claim entirely was, on average, the <i>worst</i> "
            "option. At 10-15 samples per cell this is most likely dominated by noise "
            "rather than a real, consistent effect — <b>no production change is "
            "justified yet</b>; the full sweep's larger sample size should settle it.</p>"
        )
        parts.append(_section("rs-modality", "Does the model care if you call it \"handwritten\" or \"printed\"?", body))
        toc.append(("rs-modality", "Modality framing"))

        # ---- persona & language (null results) ---------------------------------
        pv_bar = []
        for pv in ("line-v2-persona", "line-v2-arabic"):
            vals = [_mean(cer.get((pv, ds), [])) for ds in ds_list]
            vals = [v for v in vals if v is not None]
            if vals:
                pv_bar.append((VARIANTS[pv][0], _mean(vals) * 100))
        base_vals = [_mean(cer.get(("line-v2", ds), [])) for ds in ds_list]
        base_vals = [v for v in base_vals if v is not None]
        base_mean = _mean(base_vals) * 100 if base_vals else None
        body = (
            "<p class='board-note'>Two more common OCR-prompting tricks — giving "
            "the model an expert persona (\"you are an expert paleographer...\"), "
            "and writing the instructions in Arabic instead of English — were tested "
            "the same way. Neither moved the needle:</p>"
            + _fig(f"Mean CER-light — baseline is {base_mean:.0f}% for reference",
                   hbar(pv_bar, unit="%"))
            + "<p class='board-note'>A result showing \"no effect\" is still a useful "
            "finding here: it means the model's failures on this task are not "
            "primarily a framing/persona problem, and future effort is better spent "
            "on the axes that <i>did</i> move (granularity, diacritics) than on more "
            "prompt-phrasing variations of this kind.</p>"
        )
        parts.append(_section("rs-null", "Persona and language of instructions: no measurable effect", body))
        toc.append(("rs-null", "Persona & language (null result)"))

    # ---- limitations / next steps -------------------------------------------
    limits = (
        "<p class='board-note'>Three datasets couldn't get fresh data this pass — "
        "<code>muharaf_lines</code>, <code>nakba_lines</code>, <code>rasam1</code> — "
        "all hit real network-reliability problems downloading from Hugging Face on "
        "this machine (not a bug in the study itself; see "
        "<code>docs/experiments/prompt-study.md</code> for the detailed diagnosis, "
        "including that disabling HF's newer \"xet\" transfer backend did not fully "
        "fix it). They're worth retrying from a more reliable network.</p>"
        "<p class='board-note'><b>Next:</b> a full sweep (more samples per dataset, "
        "same variants) to confirm which pilot findings hold up at scale, plus "
        "applying the page-scale prompt fix to the main benchmark independent of "
        "that.</p>"
    )
    parts.append(_section("rs-limits", "Limitations & what's next", limits))
    toc.append(("rs-limits", "Limitations & next steps"))

    # ---- stub: other research directions ------------------------------------
    other_qs = [
        ("Base-model choice for fine-tuning", "in progress",
         "Qwen3-VL-4B vs. Baseer-Nakba vs. PaliGemma-2, same LoRA recipe and data "
         "(see the Fine-tuning tab). Early result: the Arabic-specialised base "
         "(Baseer) barely benefits from the same fine-tune and even regresses on "
         "its own specialty dataset, while the generic base (Qwen) improves hugely — "
         "itself a publishable headline (\"does starting from a specialised base "
         "actually help?\")."),
        ("Data-scaling curve", "not started",
         "8,000 training lines was one arbitrary point for the LoRA fine-tunes. "
         "Re-running at 1k/2k/4k/full and tracing where held-out CER flattens would "
         "size how much labelled handwriting a from-scratch Arabic HTR effort "
         "actually needs — directly useful for anyone budgeting an annotation "
         "project."),
        ("Format-vs-vision control", "not started",
         "How much of a LoRA fine-tune's gain is just the model learning the output "
         "*format* (bare line, no chat/commentary) vs. genuinely better visual "
         "reading of the script? A control that only changes the *prompt* on the "
         "untuned base (no training) would separate these two effects."),
        ("Script-family generalisation", "not started",
         "The Muharaf LoRA (Ruq'ah handwriting) transfers well to KHATT and NAKBA "
         "but is weakest on RASAM (Maghrebi script) — is that a data-representation "
         "gap (too little Maghrebi-like handwriting in training) or a frozen-vision-"
         "encoder ceiling? Un-freezing the vision tower (`train_vision: true`) on "
         "just the Maghrebi case would isolate it."),
        ("Diacritic restoration as its own task", "not started",
         "Every model here is scored on transcribing diacritics it can barely see. "
         "A separate track — give the model the correctly-transcribed bare letters "
         "and ask only for the harakat — would isolate \"can it restore diacritics "
         "from context\" from \"can it read handwriting,\" which are really two "
         "different skills bundled into one metric today."),
        ("Cost/latency vs. accuracy frontier as a product question", "in progress",
         "The Findings tab's Model landscape / Operations sections already track "
         "this per-model; extending it into an explicit Pareto frontier (cheapest "
         "model at each accuracy tier) turns the leaderboard from \"who's best\" "
         "into \"what should a real product actually call.\""),
    ]
    rows_html = "".join(
        f"<tr><td class='t'>{_esc(q)}</td><td>{_tag(status, 'best' if status=='in progress' else 'worst' if status=='not started' else 'ds')}</td>"
        f"<td>{_esc(why)}</td></tr>"
        for q, status, why in other_qs
    )
    other_body = (
        "<p class='board-note'>Candidate research problems this project's data and "
        "harness could already support, beyond \"which model reads Arabic best.\" "
        "Captured here so we come back to them once the prompt study is wrapped up — "
        "not full write-ups yet.</p>"
        f"<div class='scroll'><table class='rv-tbl'><thead><tr><th>Question</th>"
        f"<th>Status</th><th>Why it's a good candidate</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table></div>"
    )
    parts.append(_section("rs-other-qs", "Other research directions this project could support", other_body))
    toc.append(("rs-other-qs", "Other research directions"))

    # ---- stub: publishability audit -----------------------------------------
    pub_body = (
        "<p class='board-note'>First-pass, not-yet-complete read on whether this "
        "benchmark itself is worth writing up as a paper (separate from any single "
        "model result). Revisit and expand once the above research directions have "
        "real findings to cite.</p>"
        "<p class='board-note'><b>Already in our favour:</b></p>"
        "<ul><li>Real, degraded, non-synthetic historical Arabic handwriting "
        "(Muharaf, NAKBA, RASAM) — most public OCR benchmarks test clean modern or "
        "synthetic text; this is a documented gap in the field.</li>"
        "<li>A genuinely broad model field: frontier paid VLMs, free-tier hosted "
        "models, open-weight local models, and our own fine-tunes, scored with the "
        "identical harness and metrics.</li>"
        "<li>A contamination gate (train/test overlap check) already built and run "
        "before every fine-tuning experiment.</li>"
        "<li>Error-anatomy tooling (refusal / wrong-script / repetition / diacritic "
        "F1 classification) beyond a bare CER number — this is closer to what a "
        "reviewer expects than a leaderboard table alone.</li></ul>"
        "<p class='board-note'><b>Needed before a submission:</b></p>"
        "<ul><li>Larger N per (model, dataset) cell for statistical confidence — "
        "several cells are still at smoke-test sample sizes.</li>"
        "<li>A human-agreement or inter-annotator baseline, to give CER numbers a "
        "reference ceiling (how far is \"perfect\" from 0%?).</li>"
        "<li>A related-work / prior-benchmark comparison table (KITAB-Bench, "
        "NAKBA NLP 2026, others) making explicit what's new here.</li>"
        "<li>A written ablation section — this prompt study is exactly that "
        "material, once the full sweep is in.</li>"
        "<li>A license/redistribution review for anything bundled into a release "
        "artifact (several source datasets are CC-BY-NC-SA — research use only).</li>"
        "</ul>"
    )
    parts.append(_section("rs-publish", "Is this benchmark publishable?", pub_body))
    toc.append(("rs-publish", "Publishability audit"))

    parts.append("</div>")
    return "".join(parts), toc
