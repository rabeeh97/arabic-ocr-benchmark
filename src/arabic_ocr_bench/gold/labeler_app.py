"""Local labeling tool for a gold-set tranche (stdlib only; binds to 127.0.0.1).

    PYTHONPATH=src python -m arabic_ocr_bench.gold.labeler_app --tranche gold/tranches/t01 [--port 8770]

* Every labeler uses their own ID; labels go to gold/labels/<id>/<tranche>.jsonl (append-only, newest wins).
* Labelers never see each other's labels, and the toolkit's notes for a line are revealed only after that
  labeler has saved a verdict for it (so the review is not primed).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import GUIDELINE_VERSION, labels, schema, tranche

STATIC = Path(__file__).with_name("static")
REPO = Path(__file__).resolve().parents[3]


def md_to_html(md: str) -> str:
    """Tiny markdown renderer (headings, lists, bold, code, paragraphs, pipe tables as preformatted)."""
    out, in_list, in_pre = [], False, False

    def inline(s: str) -> str:
        s = escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)

    for ln in md.splitlines():
        if ln.startswith("```"):
            out.append("</pre>" if in_pre else "<pre>")
            in_pre = not in_pre
            continue
        if in_pre:
            out.append(escape(ln))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        is_li = re.match(r"^\s*[-*]\s+", ln)
        if in_list and not is_li and ln.strip():
            out.append("</ul>")
            in_list = False
        if m:
            lvl = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f"<h{lvl} id='{slug}'>{inline(m.group(2))}</h{lvl}>")
        elif is_li:
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^\s*[-*]\s+", "", ln)
            out.append(f"<li>{inline(item)}</li>")
        elif ln.startswith("|"):
            out.append(f"<div class='tbl'>{escape(ln)}</div>")
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


class App:
    def __init__(self, tranche_dir: Path, labels_root: Path, guidelines: Path | None = None):
        self.dir = tranche_dir
        self.name = tranche_dir.name
        self.labels_root = labels_root
        self.lines = tranche.load_tranche(tranche_dir)
        self.by_id = {x["line_id"]: x for x in self.lines}
        self.guidelines = guidelines

    def _view(self, x: dict, rec: dict | None, reveal: bool) -> dict:
        v = {k: x[k] for k in ("line_id", "dataset", "tier", "language", "rank", "calibration", "text_original", "tags")}
        v["has_image"] = bool(x.get("image"))
        v["label"] = rec
        v["toolkit_notes"] = x["toolkit_notes"] if reveal else None
        v["n_toolkit_notes"] = len(x["toolkit_notes"]) if reveal else None
        return v

    def state(self, labeler: str) -> dict:
        done = labels.load_labels(self.labels_root, labeler, self.name)
        return {
            "tranche": self.name, "guideline_version": GUIDELINE_VERSION, "labeler": labeler,
            "lines": [{"line_id": x["line_id"], "dataset": x["dataset"], "rank": x["rank"],
                       "calibration": x["calibration"], "verdict": (done.get(x["line_id"]) or {}).get("verdict")}
                      for x in self.lines],
            "vocab": {"verdicts": schema.VERDICTS, "correction_types": schema.CORRECTION_TYPES,
                      "image_issues": schema.IMAGE_ISSUES, "line_kinds": schema.LINE_KINDS,
                      "legibility": schema.LEGIBILITY, "diacritics_visible": schema.DIACRITICS_VISIBLE},
        }

    def line(self, line_id: str, labeler: str) -> dict | None:
        x = self.by_id.get(line_id)
        if not x:
            return None
        rec = labels.load_labels(self.labels_root, labeler, self.name).get(line_id)
        return self._view(x, rec, reveal=rec is not None)

    def save(self, labeler: str, line_id: str, form: dict) -> dict:
        x = self.by_id.get(line_id)
        if not x:
            return {"ok": False, "errors": ["unknown line"]}
        if not schema.LABELER_RE.match(labeler or ""):
            return {"ok": False, "errors": ["invalid labeler id"]}
        form = dict(form)
        form["saved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec, errs, notes = schema.new_record(x, labeler, form, x["language"])
        if errs:
            return {"ok": False, "errors": errs}
        labels.append_label(self.labels_root, labeler, self.name, rec)
        return {"ok": True, "errors": [], "hygiene": notes, "record": rec, "toolkit_notes": x["toolkit_notes"]}


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            lab = q.get("labeler", "")
            if u.path == "/":
                return self._send(200, (STATIC / "labeler.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/guidelines":
                g = app.guidelines
                body = md_to_html(g.read_text(encoding="utf-8")) if g and g.exists() else "<p>No guidelines file.</p>"
                page = ("<!doctype html><meta charset=utf-8><title>Gold set guidelines</title><style>body{font:16px/1.55 system-ui;"
                        "max-width:820px;margin:2rem auto;padding:0 1rem}code{background:#eee;padding:0 .25em;border-radius:3px}"
                        ".tbl{font-family:monospace;white-space:pre-wrap;font-size:13px}pre{background:#f4f4f4;padding:.6rem;overflow:auto}"
                        "</style>" + body)
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/state":
                if not schema.LABELER_RE.match(lab):
                    return self._json({"error": "invalid labeler id"}, 400)
                return self._json(app.state(lab))
            if u.path == "/api/line":
                v = app.line(q.get("id", ""), lab) if schema.LABELER_RE.match(lab) else None
                return self._json(v, 200) if v else self._json({"error": "not found"}, 404)
            if u.path == "/img":
                x = app.by_id.get(q.get("id", ""))
                if not x or not x.get("image"):
                    return self._send(404, b"no image", "text/plain")
                p = app.dir / x["image"]
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                return self._json({"ok": False, "errors": ["bad request"]}, 400)
            self._json(app.save(data.get("labeler", ""), data.get("line_id", ""), data.get("form", {})))

    return H


def serve(tranche_dir: str, labels_root: str, port: int = 8770, guidelines: str | None = None) -> ThreadingHTTPServer:
    app = App(Path(tranche_dir), Path(labels_root), Path(guidelines) if guidelines else REPO / "gold" / "GUIDELINES.md")
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", required=True, help="e.g. gold/tranches/t01")
    ap.add_argument("--labels", default=str(REPO / "gold" / "labels"))
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args(argv)
    srv = serve(a.tranche, a.labels, a.port)
    print(f"labeling tool for {a.tranche}: http://127.0.0.1:{a.port}/   (Ctrl+C to stop)\nlabels -> {a.labels}/<labeler-id>/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
