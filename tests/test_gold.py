import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arabic_ocr_bench.gold import agreement, images, labels, pool, schema, tags
from arabic_ocr_bench.gold.labeler_app import App, make_handler, md_to_html
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def _noise_png(seed: int, fmt: str = "PNG", size=(300, 40)) -> bytes:
    rng = np.random.default_rng(seed)
    a = (rng.random((size[1], size[0])) > 0.7).astype(np.uint8) * 255
    im = Image.fromarray(255 - a).convert("L")
    buf = io.BytesIO()
    im.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {}))
    return buf.getvalue()


# ---------------------------------------------------------------- ranking
def test_rank_is_stable_prefix_when_lines_are_added():
    ids = [pool.line_id("d", "test", i) for i in range(500)]
    order = sorted(ids, key=lambda l: pool.rank_key("seed-a", "d", l))
    more = ids + [pool.line_id("d", "test", i) for i in range(500, 700)]
    order2 = sorted(more, key=lambda l: pool.rank_key("seed-a", "d", l))
    # relative order of the old lines is unchanged
    assert [x for x in order2 if x in set(ids)] == order
    # a different seed gives a different order
    assert order != sorted(ids, key=lambda l: pool.rank_key("seed-b", "d", l))


def test_tranche_rows_prefixes_and_capacity():
    rows = [{"line_id": str(i), "rank": i} for i in range(1, 121)] + [{"line_id": "dup", "rank": None}]
    assert [r["rank"] for r in pool.tranche_rows(rows, 1, 50)] == list(range(1, 51))
    assert [r["rank"] for r in pool.tranche_rows(rows, 3, 50)] == list(range(101, 121))
    assert pool.tranche_rows(rows, 4, 50) == []


# ---------------------------------------------------------------- dedup
def _index():
    return pool.DupIndex(16, 10, 0.97, 0.05)


def test_exact_and_reencoded_duplicates_are_found_and_distinct_images_are_not():
    idx = _index()
    a, b = _noise_png(1), _noise_png(2)
    sa = images.signature(a)
    idx.add("a/0", sa)
    assert idx.find(images.signature(a)) == ("a/0", "exact")
    reenc = _noise_png(1, "JPEG")                       # same picture, re-encoded: not byte/pixel identical
    hit = idx.find(images.signature(reenc))
    assert hit is not None and hit[0] == "a/0"
    assert idx.find(images.signature(b)) is None


def test_blank_images_are_not_near_duplicates_of_each_other():
    idx = _index()
    blank1 = io.BytesIO(); Image.new("L", (200, 30), 255).save(blank1, "PNG")
    blank2 = io.BytesIO(); Image.new("L", (210, 30), 255).save(blank2, "PNG")
    idx.add("x/0", images.signature(blank1.getvalue()))
    assert idx.find(images.signature(blank2.getvalue())) is None   # different size: not exact, no correlation


def test_same_text_different_image_is_not_a_duplicate():
    idx = _index()
    idx.add("a/0", images.signature(_noise_png(10)))
    assert idx.find(images.signature(_noise_png(11))) is None


# ---------------------------------------------------------------- tags
def test_text_tags_arabic_with_and_without_marks():
    t = tags.text_tags("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", "ar")
    assert t["has_diacritics"] is True and t["diac_level"] in ("partial", "full")
    t2 = tags.text_tags("بسم الله", "ar")
    assert t2["has_diacritics"] is False and t2["diac_level"] == "none"
    assert tags.length_band(8) == "tiny_le10" and tags.length_band(60) == "medium_26_60"


def test_text_tags_digits_latin_and_english():
    t = tags.text_tags("سنة ١٩٥١ و 1951 ABC", "ar")
    assert t["has_digits"] and t["has_latin"] and set(t["digit_kinds"]) == {"arabic_indic", "western"}
    e = tags.text_tags("A move to stop Mr. Gaitskell", "en")
    assert e["script_main"] == "latin" and e["has_diacritics"] is None


def test_image_tags_have_expected_keys():
    t = images.image_tags(_noise_png(3, size=(400, 50)))
    assert t["img_h"] == 50 and t["height_tier"] == "low_le64" and 0 < t["ink_density"] < 0.5


# ---------------------------------------------------------------- schema
def test_clean_text_only_touches_hygiene():
    s, notes = schema.clean_text("  بســم‏   الله\n", "ar")
    assert s == "بسم الله" and notes
    s2, _ = schema.clean_text("a  b‎ c", "en")
    assert s2 == "a b c"


LINE = {"line_id": "d/test/000001", "dataset": "d", "tranche": 1, "text_original": "بسم الله", "language": "ar"}
BASE = {"line_kind": "main_text", "legibility": "clear", "diacritics_visible": "none"}


def _rec(form, line=LINE, labeler="rabeeh"):
    return schema.new_record(line, labeler, {**BASE, **form}, "ar")


def test_correct_sets_text_to_cleaned_original():
    rec, errs, _ = _rec({"verdict": "correct"})
    assert not errs and rec["text_verified"] == "بسم الله"


def test_corrected_requires_change_and_type():
    _, errs, _ = _rec({"verdict": "corrected", "text_verified": "بسم الله", "correction_types": ["dots"]})
    assert any("use 'correct'" in e for e in errs)
    _, errs, _ = _rec({"verdict": "corrected", "text_verified": "بسم اللة"})
    assert any("correction type" in e for e in errs)
    rec, errs, _ = _rec({"verdict": "corrected", "text_verified": "بسم اللة", "correction_types": ["letters"]})
    assert not errs and rec["text_verified"] == "بسم اللة" and rec["guideline_version"]


def test_unusable_and_unsure_need_a_reason_and_illegible_cannot_be_correct():
    _, errs, _ = _rec({"verdict": "unusable"})
    assert errs
    _, errs, _ = _rec({"verdict": "unusable", "image_issues": ["non_text"]})
    assert not errs
    _, errs, _ = _rec({"verdict": "correct", "legibility": "illegible"})
    assert any("illegible" in e for e in errs)


def test_bad_labeler_id_and_bad_vocab_rejected():
    _, errs, _ = _rec({"verdict": "correct"}, labeler="Bad Name")
    assert any("labeler" in e for e in errs)
    _, errs, _ = _rec({"verdict": "corrected", "text_verified": "x", "correction_types": ["nope"]})
    assert any("unknown correction_types" in e for e in errs)


# ---------------------------------------------------------------- labels + agreement
def test_labels_append_only_newest_wins(tmp_path):
    labels.append_label(tmp_path, "ann", "t01", {"line_id": "a", "verdict": "unsure"})
    labels.append_label(tmp_path, "ann", "t01", {"line_id": "a", "verdict": "correct"})
    assert labels.load_labels(tmp_path, "ann", "t01")["a"]["verdict"] == "correct"
    assert len((tmp_path / "ann" / "t01.jsonl").read_text().splitlines()) == 2
    assert labels.list_labelers(tmp_path, "t01") == ["ann"]


def test_agreement_report(tmp_path):
    def rec(i, lab, v, text):
        return {"line_id": i, "dataset": "d", "labeler": lab, "verdict": v, "text_verified": text}
    for i, (ta, tb, va, vb) in enumerate([("بسم", "بسم", "correct", "correct"), ("كتب", "كتاب", "corrected", "corrected"),
                                          (None, None, "unusable", "unusable"), ("ذهب", "ذهب", "correct", "corrected")]):
        labels.append_label(tmp_path, "ann", "t01", rec(f"l{i}", "ann", va, ta))
        labels.append_label(tmp_path, "bob", "t01", rec(f"l{i}", "bob", vb, tb))
    rep = agreement.compare(tmp_path, "t01")
    p = rep["pairs"][0]
    assert p["common_lines"] == 4 and p["texts_compared"] == 3
    assert p["verdict_agreement"] == 0.75 and p["exact_text"] == round(2 / 3, 3)
    assert p["biggest_disagreements"][0]["line_id"] == "l1"
    assert "ann vs bob" in agreement.to_markdown(rep)
    assert agreement.cohen_kappa(["a", "b"], ["a", "b"]) == 1.0


# ---------------------------------------------------------------- the app
@pytest.fixture()
def server(tmp_path):
    tdir = tmp_path / "t01"
    (tdir / "images").mkdir(parents=True)
    (tdir / "images" / "a.png").write_bytes(_noise_png(5))
    lines = [{"line_id": "d/test/000001", "source_id": "d/test/000001", "dataset": "d", "tier": "arabic_random",
              "language": "ar", "tranche": 1, "rank": 1, "calibration": True, "text_original": "بسم الله",
              "image": "images/a.png", "tags": {"medium": "handwritten"},
              "toolkit_notes": [{"check": "dotting_error", "level": "experimental", "detail": "x", "score": 0.4}]}]
    (tdir / "lines.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
    g = tmp_path / "G.md"
    g.write_text("# Title\n- **bold** item\n", encoding="utf-8")
    app = App(tdir, tmp_path / "labels", g)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read(), r.headers.get_content_type()


def _post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_app_blind_first_notes_and_per_labeler_files(server):
    base, tmp = server
    st = json.loads(_get(base + "/api/state?labeler=ann")[0])
    assert len(st["lines"]) == 1 and st["lines"][0]["verdict"] is None and "verdicts" in st["vocab"]
    lid = "d/test/000001"
    q = f"/api/line?id={urllib.request.quote(lid, safe='')}&labeler="
    line = json.loads(_get(base + q + "ann")[0])
    assert line["toolkit_notes"] is None                        # hidden before a verdict
    bad = _post(base + "/api/save", {"labeler": "ann", "line_id": lid, "form": {"verdict": "corrected", **BASE}})
    assert not bad["ok"]
    ok = _post(base + "/api/save", {"labeler": "ann", "line_id": lid,
                                    "form": {"verdict": "corrected", "text_verified": "بسم اللة", "correction_types": ["letters"], **BASE}})
    assert ok["ok"] and ok["toolkit_notes"][0]["check"] == "dotting_error"
    assert json.loads(_get(base + q + "ann")[0])["toolkit_notes"] is not None        # revealed after saving
    assert json.loads(_get(base + q + "bob")[0])["toolkit_notes"] is None            # other labeler still blind
    assert json.loads(_get(base + q + "bob")[0])["label"] is None                    # and cannot see ann's label
    assert (tmp / "labels" / "ann" / "t01.jsonl").exists() and not (tmp / "labels" / "bob").exists()
    body, ctype = _get(base + f"/img?id={urllib.request.quote(lid, safe='')}")
    assert ctype == "image/png" and body[:4] == b"\x89PNG"
    assert b"<h1" in _get(base + "/guidelines")[0] and b"<b>bold</b>" in _get(base + "/guidelines")[0]


def test_app_rejects_bad_labeler_and_unknown_line(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError):
        _get(base + "/api/state?labeler=Bad%20Name")
    assert not _post(base + "/api/save", {"labeler": "ann", "line_id": "nope", "form": {}})["ok"]
    assert not _post(base + "/api/save", {"labeler": "../x", "line_id": "d/test/000001", "form": {}})["ok"]


def test_md_to_html_basics():
    h = md_to_html("# A\n\ntext `c` **b**\n- x\n- y\n")
    assert "<h1" in h and "<code>c</code>" in h and h.count("<li>") == 2


# ---------------------------------------------------------------- committed pool consistency
POOL = ROOT / "gold" / "pool"


@pytest.mark.skipif(not (POOL / "_meta.json").exists(), reason="pool not built")
def test_committed_pool_is_consistent():
    cfg = pool.load_config(ROOT / "gold" / "config.yaml")
    meta = json.load(open(POOL / "_meta.json", encoding="utf-8"))
    assert meta["seed"] == cfg["seed"]
    for spec in cfg["datasets"]:
        rows = pool.read_pool(POOL, spec["key"])
        ranks = [r["rank"] for r in rows if r["rank"] is not None]
        assert ranks == list(range(1, len(ranks) + 1)), spec["key"]
        keys = [r["rank_key"] for r in rows if r["rank"] is not None]
        assert keys == sorted(keys)
        assert all(r["rank_key"] == pool.rank_key(cfg["seed"], spec["key"], r["line_id"]) for r in rows[:50])
        kept_sha = [r["sha"] for r in rows if r["rank"] is not None]
        assert len(kept_sha) == len(set(kept_sha)), "an exact duplicate image survived in " + spec["key"]
        assert meta["datasets"][spec["key"]]["eligible"] == len(ranks)
