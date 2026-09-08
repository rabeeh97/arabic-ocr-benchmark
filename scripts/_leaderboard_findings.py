"""Hand-rolled SVG / CSS chart primitives for the leaderboard's Findings view.

Standalone: no imports from ``make_leaderboard.py`` and none from the analysis
package. Every function is a pure string builder that turns plain Python
numbers / strings / lists into a well-formed HTML or SVG fragment string.
Only stdlib (``html``, ``math``).

Colour comes exclusively from the CSS custom properties defined in the
leaderboard's ``_HEAD`` block (``--ink``, ``--line``, ``--accent``,
``--accent-soft``, ``--muted``, ``--surface``, ``--ground``); the palette is
never redefined here. ``FINDINGS_CSS`` (scoped under ``.findings-view``) styles
the fragments and is wrapped in ``<style>`` by the caller. Tooltips are
declarative: elements carry a ``data-tip`` attribute the view-level JS (Task 18)
reads into the shared ``.chart-tip`` div. No ``<script>`` is emitted here.
"""

import html
import math

try:  # single-source the failure-class ordering from the analysis package
    from arabic_ocr_bench.analysis.refusal import FAILURE_CLASSES
except Exception:  # noqa: BLE001 - this scripts/ module can be imported without src on sys.path
    FAILURE_CLASSES = (
        "ok", "empty", "refusal", "meta_commentary",
        "wrong_script", "prompt_echo", "rate_limited", "other_error",
    )

__all__ = [
    "FAILURE_CLASSES", "FINDINGS_CSS", "FINDINGS_SECTIONS", "binned_curve", "box_row",
    "diverging_hbar", "dumbbell", "ecdf", "forest", "hbar", "histogram",
    "render_findings_view", "stacked_hbar", "svg_heatmap", "svg_scatter",
]


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _esc(s) -> str:
    """Escape any value for use as text or a quoted attribute value."""
    return html.escape("" if s is None else str(s), quote=True)


def _empty() -> str:
    return '<p class="chart-empty">no data</p>'


def _f(v) -> float:
    """Coerce to a finite float, falling back to 0.0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) or math.isinf(f) else f


def _num(v, d: int = 2) -> str:
    """Compact fixed-point string (trailing zeros trimmed)."""
    s = f"{_f(v):.{d}f}".rstrip("0").rstrip(".")
    return s or "0"


def _svg(inner: str, w: float, h: float, cls: str) -> str:
    return (
        f'<svg class="fv-svg {cls}" viewBox="0 0 {w:g} {h:g}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>'
    )


_DASH = ("0", "6 3", "2 3", "9 3 2 3", "5 4", "1 3")


def _sstyle(i: int) -> str:
    """Per-series stroke presentation (opacity + dash) over the single accent."""
    op = 1.0 - 0.14 * (i % 4)
    return f'stroke="var(--accent)" stroke-opacity="{op:.2f}" stroke-dasharray="{_DASH[i % len(_DASH)]}"'


def _seg_colors(colors, n: int) -> list:
    """CSS colour per stacked segment. Default: cycle the five theme-aware
    ``--fb-seg-1..5`` vars (defined in ``FINDINGS_CSS``). ``colors`` overrides:
    a list of ``--var`` names or 1-based seg indices, cycled to length ``n``."""
    if not colors:
        return [f"var(--fb-seg-{(i % 5) + 1})" for i in range(n)]
    out = []
    for i in range(n):
        c = colors[i % len(colors)]
        if isinstance(c, bool):
            c = 1
        if isinstance(c, int):
            out.append(f"var(--fb-seg-{((c - 1) % 5) + 1})")
        else:
            c = str(c)
            out.append(f"var({c})" if c.startswith("--") else c)
    return out


# --------------------------------------------------------------------------- #
# CSS                                                                         #
# --------------------------------------------------------------------------- #
FINDINGS_CSS = """
.findings-view{--fb-h:22px;color:var(--ink);font-size:13px;
  --fb-seg-1:var(--accent);
  --fb-seg-2:color-mix(in srgb, var(--accent) 50%, var(--muted));
  --fb-seg-3:var(--muted);
  --fb-seg-4:color-mix(in srgb, var(--muted) 55%, var(--ground));
  --fb-seg-5:color-mix(in srgb, var(--accent) 42%, var(--ground));}
.findings-view .chart-empty{margin:8px 2px;padding:14px;border:1px dashed var(--line);
  border-radius:10px;background:var(--surface);color:var(--muted);font-size:12px;text-align:center;}
.findings-view .chart-tip{position:fixed;z-index:60;left:0;top:0;pointer-events:none;opacity:0;
  max-width:280px;padding:6px 9px;border:1px solid var(--line);border-radius:8px;
  background:var(--surface);color:var(--ink);font-size:11.5px;line-height:1.45;
  box-shadow:0 4px 16px color-mix(in srgb, var(--ink) 16%, transparent);transition:opacity .1s;}
.findings-view .chart-tip.on{opacity:1;}
.findings-view figure{margin:0 0 12px;}
.findings-view figcaption{margin:0 0 6px;color:var(--muted);font-size:11px;
  font-family:"IBM Plex Mono",monospace;letter-spacing:.02em;}
.findings-view .fv-svg{display:block;max-width:100%;overflow:visible;}
.findings-view .fv-svg text{fill:var(--muted);font-size:10px;font-family:"IBM Plex Mono",monospace;}
.findings-view .fv-svg .fv-title{fill:var(--ink);font-size:11px;}
.findings-view .fv-svg .ax{stroke:var(--line);stroke-width:1;}
.findings-view .fv-svg .hm-cell{stroke:var(--surface);stroke-width:1.5;}
.findings-view .fv-svg .pt{fill:var(--accent);fill-opacity:.5;}
.findings-view .fv-svg .pt-fr{fill:var(--accent);}
.findings-view .fv-svg .pt-solid{fill:var(--accent);}
.findings-view .fv-svg .ln{fill:none;stroke:var(--accent);stroke-width:1.6;}
.findings-view .fv-svg .frontier{fill:none;stroke:var(--accent);stroke-width:1.4;stroke-dasharray:4 3;}
.findings-view .fv-svg .bar{fill:var(--accent);fill-opacity:.75;}
.findings-view .fv-svg .whisk{stroke:var(--muted);stroke-width:1.4;}
.findings-view .fv-svg .box{fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.2;}
.findings-view .fv-svg .med{stroke:var(--accent);stroke-width:2;}
.findings-view .fv-svg .db-l{fill:var(--muted);}
.findings-view .fv-svg .db-r{fill:var(--accent);}
.findings-view .fv-svg .db-conn{stroke:var(--line);stroke-width:2;}
.findings-view .fb{margin:4px 0 8px;}
.findings-view .fb-row{display:flex;align-items:center;gap:8px;margin:3px 0;}
.findings-view .fb-lab{flex:none;width:132px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;text-align:right;color:var(--muted);font-size:11.5px;}
.findings-view .fb-lab.rtl{direction:rtl;text-align:left;}
.findings-view .fb-track,.findings-view .fb-div{position:relative;flex:1;height:var(--fb-h);
  border-radius:5px;background:var(--accent-soft);overflow:hidden;}
.findings-view .fb-fill{position:absolute;top:0;bottom:0;left:0;background:var(--accent);border-radius:5px;}
.findings-view .fb-seg{position:absolute;top:0;bottom:0;}
.findings-view .fb-div-mid{position:absolute;top:0;bottom:0;left:50%;width:1px;background:var(--line);}
.findings-view .fb-val{flex:none;width:64px;color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums;}
.findings-view .fb-legend{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;color:var(--muted);font-size:11px;}
.findings-view .fb-legend i{display:inline-block;width:10px;height:10px;margin-right:4px;
  border-radius:2px;vertical-align:-1px;background:var(--accent);}
"""


# --------------------------------------------------------------------------- #
# primitives                                                                  #
# --------------------------------------------------------------------------- #
def svg_heatmap(rows, cols, values, *, fmt="{:.0f}", title="", rtl_labels=False) -> str:
    if not rows or not cols or not values:
        return _empty()
    flat = [c for r in values for c in r if isinstance(c, (int, float))]
    lo, hi = (min(flat), max(flat)) if flat else (0.0, 1.0)
    rng = (hi - lo) or 1.0
    cw, ch = 46, 26
    left, top = 128, (30 if title else 16)
    w, h = left + cw * len(cols) + 8, top + ch * len(rows) + 34
    p = []
    if title:
        p.append(f'<text x="{left}" y="14" class="fv-title">{_esc(title)}</text>')
    for ci, cn in enumerate(cols):
        p.append(f'<text x="{left + ci*cw + cw/2:g}" y="{top-4:g}" text-anchor="middle">{_esc(cn)}</text>')
    for ri, rn in enumerate(rows):
        y = top + ri * ch
        lx, anc = (8, "start") if rtl_labels else (left - 8, "end")
        p.append(f'<text x="{lx:g}" y="{y + ch/2 + 3:g}" text-anchor="{anc}">{_esc(rn)}</text>')
        row = values[ri] if ri < len(values) else []
        for ci, cn in enumerate(cols):
            v = row[ci] if ci < len(row) else None
            x = left + ci * cw
            if isinstance(v, (int, float)):
                op = 0.08 + 0.85 * (v - lo) / rng
                p.append(
                    f'<rect class="hm-cell" x="{x:g}" y="{y:g}" width="{cw}" height="{ch}" '
                    f'fill="var(--accent)" fill-opacity="{op:.3f}" '
                    f'data-tip="{_esc(f"{rn} / {cn}: " + fmt.format(v))}"></rect>'
                    f'<text x="{x + cw/2:g}" y="{y + ch/2 + 3:g}" text-anchor="middle" '
                    f'fill="var(--ink)">{_esc(fmt.format(v))}</text>'
                )
            else:
                p.append(f'<rect class="hm-cell" x="{x:g}" y="{y:g}" width="{cw}" height="{ch}" '
                         f'fill="var(--surface)"></rect>')
    return _svg("".join(p), w, h, "fv-heatmap")


def hbar(pairs, *, unit="", max_bars=30, rtl_labels=False) -> str:
    pairs = list(pairs)[: max(1, int(max_bars))]
    if not pairs:
        return _empty()
    vals = [_f(v) for _, v in pairs]
    hi = max(vals + [0.0]) or 1.0
    rtl = " rtl" if rtl_labels else ""
    out = ['<div class="fb">']
    for (lab, v), fv in zip(pairs, vals):
        wd = max(0.0, min(100.0, fv / hi * 100.0))
        out.append(
            f'<div class="fb-row"><span class="fb-lab{rtl}" title="{_esc(lab)}">{_esc(lab)}</span>'
            f'<span class="fb-track"><span class="fb-fill" style="width:{wd:.2f}%" '
            f'data-tip="{_esc(f"{lab}: {_num(v)}{unit}")}"></span></span>'
            f'<span class="fb-val">{_esc(f"{_num(v)}{unit}")}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


def stacked_hbar(rows, segments, *, colors=None, normalize=False) -> str:
    rows, segments = list(rows), list(segments)
    if not rows or not segments:
        return _empty()
    cols = _seg_colors(colors, len(segments))
    tot_max = max((sum(_f(d.get(s, 0)) for s in segments) for _, d in rows), default=0.0) or 1.0
    leg = "".join(
        f'<span><i style="background:{cols[i]}"></i>{_esc(s)}</span>' for i, s in enumerate(segments)
    )
    out = [f'<div class="fb-legend">{leg}</div><div class="fb">']
    for lab, d in rows:
        vs = [_f(d.get(s, 0)) for s in segments]
        base = (sum(vs) or 1.0) if normalize else tot_max
        x, segs = 0.0, []
        for i, (s, v) in enumerate(zip(segments, vs)):
            wd = v / base * 100.0
            segs.append(
                f'<span class="fb-seg" style="left:{x:.2f}%;width:{wd:.2f}%;'
                f'background:{cols[i]}" '
                f'data-tip="{_esc(f"{lab} / {s}: {_num(v)}")}"></span>'
            )
            x += wd
        out.append(
            f'<div class="fb-row"><span class="fb-lab" title="{_esc(lab)}">{_esc(lab)}</span>'
            f'<span class="fb-track">{"".join(segs)}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


def diverging_hbar(rows, *, left_key, right_key, left_label, right_label) -> str:
    rows = list(rows)
    if not rows:
        return _empty()
    mx = max(
        (max(_f(d.get(left_key, 0)), _f(d.get(right_key, 0))) for _, d in rows), default=0.0
    ) or 1.0
    out = [
        (f'<div class="fb-legend"><span><i style="background:var(--muted)"></i>{_esc(left_label)}</span>'
         f'<span><i></i>{_esc(right_label)}</span></div><div class="fb">'),
    ]
    for lab, d in rows:
        lv, rv = _f(d.get(left_key, 0)), _f(d.get(right_key, 0))
        out.append(
            f'<div class="fb-row"><span class="fb-lab" title="{_esc(lab)}">{_esc(lab)}</span>'
            f'<span class="fb-div"><span class="fb-div-mid"></span>'
            f'<span class="fb-seg" style="right:50%;width:{lv / mx * 50:.2f}%;background:var(--muted)" '
            f'data-tip="{_esc(f"{lab} / {left_label}: {_num(lv)}")}"></span>'
            f'<span class="fb-seg" style="left:50%;width:{rv / mx * 50:.2f}%;background:var(--accent)" '
            f'data-tip="{_esc(f"{lab} / {right_label}: {_num(rv)}")}"></span>'
            f'</span></div>'
        )
    out.append("</div>")
    return "".join(out)


def svg_scatter(points, *, x_label, y_label, pareto=False, log_x=False, labels=None) -> str:
    pts = [(_f(p[0]), _f(p[1])) for p in points]
    if not pts:
        return _empty()
    W, H, ml, mr, mt, mb = 640, 380, 54, 16, 14, 40
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    gut = any(x <= 0 for x in xs)  # zero/negative-x points get a left "$0" gutter (both scales)
    pos = [x for x in xs if x > 0] or [1.0]
    x0 = ml + (26 if gut else 0)
    lglo, lghi = math.log10(min(pos)), math.log10(max(pos) if max(pos) > min(pos) else min(pos) * 10)
    xlo, xhi = (min(pos) if gut else min(xs)), max(xs)
    ylo, yhi = min(ys), max(ys)

    def fx(x):
        if x <= 0:
            return ml + 12
        if log_x:
            return x0 + (W - mr - x0) * (math.log10(x) - lglo) / ((lghi - lglo) or 1)
        return x0 + (W - mr - x0) * (x - xlo) / ((xhi - xlo) or 1)

    def fy(y):
        return (H - mb) - (H - mb - mt) * (y - ylo) / ((yhi - ylo) or 1)

    p = [
        f'<line class="ax" x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}"></line>',
        f'<line class="ax" x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}"></line>',
        f'<text x="{(ml+W-mr)/2:g}" y="{H-6}" text-anchor="middle">{_esc(x_label)}</text>',
        (f'<text x="12" y="{(mt+H-mb)/2:g}" text-anchor="middle" '
         f'transform="rotate(-90 12 {(mt+H-mb)/2:g})">{_esc(y_label)}</text>'),
    ]
    if gut:
        p.append(f'<text x="{ml+12:g}" y="{H-mb+14}" text-anchor="middle">$0</text>')
    front = set()
    if pareto:
        best = float("inf")
        poly = []
        for i in sorted(range(len(pts)), key=lambda k: (pts[k][0], pts[k][1])):
            if pts[i][1] <= best:
                best = pts[i][1]
                front.add(i)
                poly.append(f"{fx(pts[i][0]):g},{fy(pts[i][1]):g}")
        if len(poly) > 1:
            p.append(f'<polyline class="frontier" points="{" ".join(poly)}"></polyline>')
    for i, (x, y) in enumerate(pts):
        tip = ""
        if labels and i < len(labels) and labels[i]:
            tip = f' data-tip="{_esc(labels[i])}"'
        cls, r = ("pt-fr", 4) if i in front else ("pt", 3)
        p.append(f'<circle class="{cls}" cx="{fx(x):g}" cy="{fy(y):g}" r="{r}"{tip}></circle>')
    return _svg("".join(p), W, H, "fv-scatter")


def binned_curve(series, *, x_label, y_label) -> str:
    series = {k: list(v) for k, v in (series or {}).items() if v}
    if not series:
        return _empty()
    W, H, ml, mr, mt, mb = 640, 340, 54, 14, 12, 38
    pts = [(p[0], p[1]) for s in series.values() for p in s]
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys + [0.0]), max(ys)

    def fx(x):
        return ml + (W - ml - mr) * (_f(x) - xlo) / ((xhi - xlo) or 1)

    def fy(y):
        return (H - mb) - (H - mb - mt) * (_f(y) - ylo) / ((yhi - ylo) or 1)

    p = [
        f'<line class="ax" x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}"></line>',
        f'<line class="ax" x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}"></line>',
        f'<text x="{(ml+W-mr)/2:g}" y="{H-6}" text-anchor="middle">{_esc(x_label)}</text>',
        (f'<text x="12" y="{(mt+H-mb)/2:g}" text-anchor="middle" '
         f'transform="rotate(-90 12 {(mt+H-mb)/2:g})">{_esc(y_label)}</text>'),
    ]
    for i, (name, raw) in enumerate(sorted(series.items())):
        sp = sorted(raw, key=lambda q: q[0])
        p.append(
            f'<polyline class="ln" points="{" ".join(f"{fx(x):g},{fy(y):g}" for x, y, *_ in sp)}" '
            f'{_sstyle(i)}></polyline>'
        )
        for x, y, *n in sp:
            tip = _esc(f"{name} / {_num(x)}: {_num(y)}" + (f" (n={int(n[0])})" if n else ""))
            p.append(f'<circle class="pt-solid" cx="{fx(x):g}" cy="{fy(y):g}" r="2.5" data-tip="{tip}"></circle>')
    return _svg("".join(p), W, H, "fv-binned")


def histogram(values, *, bins=20, x_label="") -> str:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return _empty()
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    nb = max(1, int(bins))
    step = (hi - lo) / nb
    counts = [0] * nb
    for v in vals:
        counts[min(nb - 1, int((v - lo) / step))] += 1
    cmax = max(counts) or 1
    W, H, ml, mr, mt, mb = 640, 300, 44, 12, 12, 36
    bw = (W - ml - mr) / nb
    p = [
        f'<line class="ax" x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}"></line>',
        f'<line class="ax" x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}"></line>',
    ]
    for i, c in enumerate(counts):
        bh = (H - mb - mt) * c / cmax
        tip = _esc(f"[{_num(lo + i*step, 3)}, {_num(lo + (i+1)*step, 3)}): {c}")
        p.append(
            f'<rect class="bar" x="{ml + i*bw:g}" y="{H-mb-bh:g}" width="{max(0.5, bw-1):g}" '
            f'height="{bh:g}" data-tip="{tip}"></rect>'
        )
    if x_label:
        p.append(f'<text x="{(ml+W-mr)/2:g}" y="{H-6}" text-anchor="middle">{_esc(x_label)}</text>')
    return _svg("".join(p), W, H, "fv-hist")


def ecdf(series, *, x_label="CER") -> str:
    series = {k: list(v) for k, v in (series or {}).items() if v}
    if not series:
        return _empty()
    W, H, ml, mr, mt, mb = 640, 320, 48, 60, 12, 36
    allx = [x for s in series.values() for x, _ in s]
    xlo, xhi = min(allx), max(allx)

    def fx(x):
        return ml + (W - ml - mr) * (_f(x) - xlo) / ((xhi - xlo) or 1)

    def fy(y):
        return (H - mb) - (H - mb - mt) * max(0.0, min(1.0, _f(y)))

    p = [
        f'<line class="ax" x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}"></line>',
        f'<line class="ax" x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}"></line>',
        f'<text x="{(ml+W-mr)/2:g}" y="{H-6}" text-anchor="middle">{_esc(x_label)}</text>',
    ]
    for i, (name, raw) in enumerate(sorted(series.items())):
        sp = sorted(raw, key=lambda q: q[0])
        step, px, py = [], None, None
        for x, y in sp:
            if px is not None:
                step.append(f"{fx(x):g},{fy(py):g}")
            step.append(f"{fx(x):g},{fy(y):g}")
            px, py = x, y
        p.append(f'<polyline class="ln" points="{" ".join(step)}" {_sstyle(i)}></polyline>')
        p.append(
            f'<text x="{W-mr+4:g}" y="{fy(sp[-1][1]):g}" data-tip="{_esc(name)}">{_esc(name)}</text>'
        )
    return _svg("".join(p), W, H, "fv-ecdf")


def box_row(rows, *, value_range=None) -> str:
    rows = list(rows)
    if not rows:
        return _empty()
    keys = ("min", "q1", "median", "q3", "max")
    allv = [_f(d.get(k, 0)) for _, d in rows for k in keys]
    lo, hi = value_range if value_range else (min(allv), max(allv))
    if hi <= lo:
        hi = lo + 1.0
    W, rh, ml, mr = 640, 30, 132, 16
    H = 16 + rh * len(rows)

    def fx(v):
        return ml + (W - ml - mr) * (_f(v) - lo) / (hi - lo)

    p = []
    for i, (lab, d) in enumerate(rows):
        cy = 16 + rh * i + rh / 2 - 6
        b = {k: _f(d.get(k, 0)) for k in keys}
        tip = _esc(
            f"{lab}: min {_num(b['min'], 3)} q1 {_num(b['q1'], 3)} med {_num(b['median'], 3)} "
            f"q3 {_num(b['q3'], 3)} max {_num(b['max'], 3)}"
        )
        p.append(
            f'<text x="{ml-8}" y="{cy+4:g}" text-anchor="end">{_esc(lab)}</text>'
            f'<line class="whisk" x1="{fx(b["min"]):g}" y1="{cy:g}" x2="{fx(b["q1"]):g}" y2="{cy:g}"></line>'
            f'<line class="whisk" x1="{fx(b["q3"]):g}" y1="{cy:g}" x2="{fx(b["max"]):g}" y2="{cy:g}"></line>'
            f'<rect class="box" x="{fx(b["q1"]):g}" y="{cy-8:g}" '
            f'width="{max(0.5, fx(b["q3"]) - fx(b["q1"])):g}" height="16" data-tip="{tip}"></rect>'
            f'<line class="med" x1="{fx(b["median"]):g}" y1="{cy-8:g}" x2="{fx(b["median"]):g}" y2="{cy+8:g}"></line>'
        )
    return _svg("".join(p), W, H, "fv-box")


def dumbbell(rows, *, left_label, right_label) -> str:
    rows = list(rows)
    if not rows:
        return _empty()
    allv = [v for _, a, b in rows for v in (_f(a), _f(b))]
    lo, hi = min(allv), max(allv)
    if hi <= lo:
        hi = lo + 1.0
    W, rh, ml, mr = 640, 30, 132, 20
    H = 30 + rh * len(rows)

    def fx(v):
        return ml + (W - ml - mr) * (_f(v) - lo) / (hi - lo)

    p = [f'<text x="{ml}" y="12">{_esc(left_label)} &#8594; {_esc(right_label)}</text>']
    for i, (lab, a, b) in enumerate(rows):
        cy = 26 + rh * i + rh / 2
        av, bv = _f(a), _f(b)
        p.append(
            f'<text x="{ml-8}" y="{cy+4:g}" text-anchor="end">{_esc(lab)}</text>'
            f'<line class="db-conn" x1="{fx(av):g}" y1="{cy:g}" x2="{fx(bv):g}" y2="{cy:g}"></line>'
            f'<circle class="db-l" cx="{fx(av):g}" cy="{cy:g}" r="4.5" '
            f'data-tip="{_esc(f"{lab} / {left_label}: {_num(av, 3)}")}"></circle>'
            f'<circle class="db-r" cx="{fx(bv):g}" cy="{cy:g}" r="4.5" '
            f'data-tip="{_esc(f"{lab} / {right_label}: {_num(bv, 3)}")}"></circle>'
        )
    return _svg("".join(p), W, H, "fv-dumbbell")


def forest(rows) -> str:
    rows = list(rows)
    if not rows:
        return _empty()
    allv = [v for _, pt, clo, chi in rows for v in (_f(pt), _f(clo), _f(chi))]
    lo, hi = min(allv), max(allv)
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    W, rh, ml, mr = 640, 28, 140, 16
    H = 16 + rh * len(rows)

    def fx(v):
        return ml + (W - ml - mr) * (_f(v) - lo) / (hi - lo)

    p = []
    for i, (lab, pt, clo, chi) in enumerate(rows):
        cy = 16 + rh * i + rh / 2 - 6
        pv, lv, hv = _f(pt), _f(clo), _f(chi)
        tip = _esc(f"{lab}: {_num(pv, 3)} [{_num(lv, 3)}, {_num(hv, 3)}]")
        p.append(
            f'<text x="{ml-8}" y="{cy+4:g}" text-anchor="end">{_esc(lab)}</text>'
            f'<line class="whisk" x1="{fx(lv):g}" y1="{cy:g}" x2="{fx(hv):g}" y2="{cy:g}"></line>'
            f'<line class="whisk" x1="{fx(lv):g}" y1="{cy-4:g}" x2="{fx(lv):g}" y2="{cy+4:g}"></line>'
            f'<line class="whisk" x1="{fx(hv):g}" y1="{cy-4:g}" x2="{fx(hv):g}" y2="{cy+4:g}"></line>'
            f'<circle class="pt-solid" cx="{fx(pv):g}" cy="{cy:g}" r="4" data-tip="{tip}"></circle>'
        )
    return _svg("".join(p), W, H, "fv-forest")

# --------------------------------------------------------------------------- #
# section renderers                                                           #
# --------------------------------------------------------------------------- #
# ``(slug, label)`` pairs in render order. Coverage is first: it gates an
# honest reading of every section that follows.
FINDINGS_SECTIONS = [
    ("coverage", "Coverage"),
    ("error-anatomy", "Error anatomy"),
    ("script-letters", "Script & letters"),
    ("diacritics", "Diacritics"),
    ("difficulty", "Difficulty"),
    ("model-landscape", "Model landscape"),
    ("operational", "Operations"),
    ("corpus-health", "Corpus health"),
]

_NA = ('<p class="chart-empty">not available &#8212; run '
       '<code>bench analyze</code></p>')


def _panel(slug: str, title: str, note: str, body: str) -> str:
    """The section shell every ``_render_*`` returns, matching the leaderboard
    section contract (``<section id=... class='f-panel'><h2>…</h2>…``)."""
    n = f'<p class="board-note">{_esc(note)}</p>' if note else ""
    return (f"<section id='f-{_esc(slug)}' class='f-panel'>"
            f"<h2>{_esc(title)}</h2>{n}{body}</section>")


def _fig(caption: str, chart: str) -> str:
    return f'<figure><figcaption>{_esc(caption)}</figcaption>{chart}</figure>'


def _tbl(headers, rows) -> str:
    """A ``.rv-tbl`` inside a ``.scroll`` wrapper (classes from the main
    stylesheet). Every cell is escaped."""
    if not rows:
        return _NA
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return (f"<div class='scroll'><table class='rv-tbl'><thead><tr>{head}</tr>"
            f"</thead><tbody>{body}</tbody></table></div>")


def _pct(v, d: int = 0) -> str:
    return f"{_f(v) * 100:.{d}f}%"


def _models(findings: dict) -> list:
    return list((findings.get("meta") or {}).get("models") or [])


def _render_coverage(findings: dict) -> str:
    note = ("Every model and dataset: lines scored divided by lines attempted. "
            "Read this panel first — a model with thin coverage on a dataset "
            "can look artificially strong or weak everywhere else.")
    cov = (findings.get("operational") or {}).get("coverage") or {}
    grid = cov.get("grid") or []
    models, datasets = cov.get("models") or [], cov.get("datasets") or []
    if not grid or not models or not datasets:
        return _panel("coverage", "Coverage", note, _NA)
    vals = []
    for mi in range(len(models)):
        row = []
        mrow = grid[mi] if mi < len(grid) else []
        for di in range(len(datasets)):
            cell = mrow[di] if di < len(mrow) else None
            if isinstance(cell, (list, tuple)) and len(cell) == 2 and _f(cell[1]):
                row.append(_f(cell[0]) / _f(cell[1]))
            else:
                row.append(None)
        vals.append(row)
    gen = (findings.get("meta") or {}).get("generated_utc")
    body = _fig("scored / attempted",
                svg_heatmap(models, datasets, vals, fmt="{:.0%}",
                            title="scored / attempted"))
    if gen:
        body += f'<p class="board-note">Generated {_esc(gen)}.</p>'
    return _panel("coverage", "Coverage", note, body)


def _render_error_anatomy(findings: dict) -> str:
    note = ("Where the edit distance goes: substitutions vs insertions vs "
            "deletions, whether errors are only spacing, whether predictions run "
            "long or short, and the most common character rewrites.")
    ea = findings.get("error_anatomy") or {}
    if not ea:
        return _panel("error-anatomy", "Error anatomy", note, _NA)
    body = []

    ss = ea.get("sid_split") or {}
    if ss:
        order = (["__pooled__"] if "__pooled__" in ss else []) + \
                sorted(k for k in ss if k != "__pooled__")
        rows = [("pooled" if k == "__pooled__" else k, ss[k]) for k in order]
        body.append(_fig("S / I / D share of total edit distance",
                         stacked_hbar(rows, ["sub", "ins", "del"])))

    lb = ea.get("length_bias") or {}
    if lb:
        rows = []
        for m in sorted(lb):
            dm = _f((lb[m] or {}).get("delta_median"))
            rows.append((m, {"shorter": max(0.0, -dm), "longer": max(0.0, dm)}))
        body.append(_fig("length bias — median len(pred) − len(GT)",
                         diverging_hbar(rows, left_key="shorter",
                                        right_key="longer",
                                        left_label="pred shorter",
                                        right_label="pred longer")))

    ch = (ea.get("rewrites") or {}).get("char") or []
    if ch:
        pairs = [(f"{a} → {b}", c) for a, b, c in ch]
        body.append(_fig("top character rewrites (GT → pred)",
                         hbar(pairs, rtl_labels=True)))

    sp = ea.get("spacing_share") or {}
    if sp:
        rows = [(m, f"{_f(d.get('wer_light')):.2f}", d.get("spacing_ops", 0),
                 _pct(d.get("spacing_frac_of_ops")))
                for m, d in sorted(sp.items())]
        body.append(_fig("spacing-only operations",
                         _tbl(["model", "WER-light", "spacing ops",
                               "spacing % of ops"], rows)))

    return _panel("error-anatomy", "Error anatomy", note,
                  "".join(body) or _NA)


def _render_script_letters(findings: dict) -> str:
    note = ("Letter-level behaviour: which characters get confused for which, "
            "per-letter error rates, the rasm-vs-full CER gap (missing dots), and "
            "leakage of presentation forms or digit scripts.")
    sl = findings.get("script_letters") or {}
    if not sl:
        return _panel("script-letters", "Script & letters", note, _NA)
    body = []

    pooled = (sl.get("confusion") or {}).get("pooled") or []
    if pooled:
        chars = sorted({c for pair in pooled for c in (pair[0], pair[1])})
        cmap = {(a, b): n for a, b, n in pooled}
        vals = [[cmap.get((r, c)) for c in chars] for r in chars]
        body.append(_fig("substitution pairs (count)",
                         svg_heatmap(chars, chars, vals,
                                     title="GT (row) → pred (col)",
                                     rtl_labels=True)))

    ler = (sl.get("letter_error_rate") or {}).get("pooled") or []
    if ler:
        body.append(_fig("per-letter error rate (worst first)",
                         hbar([(row[0], row[3]) for row in ler],
                              rtl_labels=True)))

    rg = sl.get("rasm_gap") or {}
    if rg:
        rows = []
        for m in sorted(rg):
            for ds in sorted(rg[m] or {}):
                d = rg[m][ds]
                rows.append((f"{m} / {ds}", d.get("cer_rasm"),
                             d.get("cer_light")))
        body.append(_fig("rasm vs full CER (i'jam gap)",
                         dumbbell(rows, left_label="rasm",
                                  right_label="light")))

    pl = sl.get("presentation_leak") or {}
    if pl:
        body.append(_fig("presentation-form / tatweel leakage rate",
                         hbar([(m, (pl[m] or {}).get("rate"))
                               for m in sorted(pl)])))

    dsw = sl.get("digit_swaps") or {}
    if dsw:
        body.append(_fig("digit-script swaps",
                         hbar([(m, dsw[m]) for m in sorted(dsw)])))

    return _panel("script-letters", "Script & letters", note,
                  "".join(body) or _NA)


def _render_diacritics(findings: dict) -> str:
    note = ("How each model handles Arabic diacritics: marks missed vs inserted, "
            "diacritic F1 against CER-light, and how densely the ground truth "
            "itself is marked.")
    di = findings.get("diacritics") or {}
    if not di:
        return _panel("diacritics", "Diacritics", note, _NA)
    body = []

    per_mark = di.get("per_mark") or {}
    agg = {}
    for marks in per_mark.values():
        for mk, d in (marks or {}).items():
            a = agg.setdefault(mk, {"missed": 0.0, "inserted": 0.0})
            a["missed"] += _f(d.get("missed"))
            a["inserted"] += _f(d.get("inserted"))
    if agg:
        rows = [(mk, agg[mk]) for mk in sorted(agg)]
        body.append(_fig("per mark — missed vs inserted (pooled)",
                         diverging_hbar(rows, left_key="missed",
                                        right_key="inserted",
                                        left_label="missed",
                                        right_label="inserted")))

    fc = di.get("f1_vs_cer") or {}
    if fc:
        ms = sorted(fc)
        pts = [((fc[m] or {}).get("diac_f1"), (fc[m] or {}).get("cer_light"))
               for m in ms]
        body.append(_fig("diacritic F1 vs CER-light",
                         svg_scatter(pts, x_label="diacritic F1",
                                     y_label="CER-light", labels=ms)))

    gd = di.get("gt_density") or {}
    if gd:
        chips = ", ".join(f"{k} {_pct(v, 1)}" for k, v in sorted(gd.items()))
        body.append(f'<p class="board-note">GT diacritization density: '
                    f'{_esc(chips)}.</p>')

    return _panel("diacritics", "Diacritics", note, "".join(body) or _NA)


def _render_difficulty(findings: dict) -> str:
    note = ("Which lines are hard: consensus-hard and consensus-easy lines, the "
            "most discriminative lines, suspected bad ground truth, and how CER "
            "varies with line length.")
    df = findings.get("difficulty") or {}
    if not df:
        return _panel("difficulty", "Difficulty", note, _NA)
    models = _models(findings)
    body = []

    def _sample_tbl(items, extra=()):
        if not items:
            return _NA
        ms = models or sorted(
            {k for it in items for k in (it.get("per_model") or {})})
        headers = (["sample", "dataset", "GT"] + [f"CER {m}" for m in ms] +
                   ["mean CER"] + [h for h, _ in extra])
        rows = []
        for it in items:
            pm = it.get("per_model") or {}
            cells = [it.get("sample_id"), it.get("dataset"), it.get("gt")]
            for m in ms:
                v = pm.get(m)
                cells.append("—" if v is None else _num(v, 3))
            cells.append(_num(it.get("mean_cer"), 3))
            for _, key in extra:
                v = it.get(key)
                cells.append(_num(v, 3) if isinstance(v, (int, float))
                             else (v or ""))
            rows.append(cells)
        return _tbl(headers, rows)

    body.append(_fig("consensus-hard lines (highest mean CER)",
                     _sample_tbl(df.get("consensus_hard") or [])))
    body.append(_fig("consensus-easy lines (near-zero for every model)",
                     _sample_tbl(df.get("consensus_easy") or [])))
    body.append(_fig("discriminative lines (highest cross-model variance)",
                     _sample_tbl(df.get("discriminative") or [],
                                 extra=[("variance", "variance")])))

    sbg = df.get("suspected_bad_gt") or []
    if sbg:
        rows = [(it.get("sample_id"), it.get("dataset"), it.get("gt"),
                 _num(it.get("mean_pairwise_pred_cer"), 3),
                 _num(it.get("mean_pred_vs_gt_cer"), 3),
                 _num(it.get("score"), 3)) for it in sbg]
        body.append(_fig("suspected bad ground truth (models agree, GT disagrees)",
                         _tbl(["sample", "dataset", "GT", "pairwise pred CER",
                               "pred vs GT CER", "score"], rows)))

    cvl = df.get("cer_vs_length") or {}
    series = {m: v for m, v in cvl.items() if v}
    if series:
        body.append(_fig("CER vs GT length (binned)",
                         binned_curve(series, x_label="GT length (chars)",
                                      y_label="CER-light")))

    cd = df.get("cer_distribution") or {}
    rows = []
    for m in sorted(cd):
        for ds in sorted(cd[m] or {}):
            rows.append((f"{m} / {ds}", cd[m][ds]))
    if rows:
        body.append(_fig("per-sample CER distribution (model × dataset)",
                         box_row(rows)))

    ec = {m: v for m, v in (df.get("ecdf") or {}).items() if v}
    if ec:
        body.append(_fig("CER ECDF per model",
                         ecdf(ec, x_label="CER-light")))

    return _panel("difficulty", "Difficulty", note, "".join(body) or _NA)


def _render_model_landscape(findings: dict) -> str:
    note = ("How the models relate: output agreement, head-to-head win rate, "
            "metric correlation, ensemble ceilings, bootstrap confidence "
            "intervals, rank stability, and the normalization-profile split.")
    ml = findings.get("model_landscape") or {}
    if not ml:
        return _panel("model-landscape", "Model landscape", note, _NA)
    body = []

    ag = ml.get("agreement") or {}
    am = ag.get("models") or []
    if am and ag.get("identity"):
        body.append(_fig("pairwise output-identity rate",
                         svg_heatmap(am, am, ag.get("identity"),
                                     fmt="{:.0%}",
                                     title="output-identity rate")))

    wr = ml.get("win_rate") or {}
    wm = wr.get("models") or []
    if wm and wr.get("matrix"):
        body.append(_fig("pairwise win rate (row beats column)",
                         svg_heatmap(wm, wm, wr.get("matrix"), fmt="{:.0%}",
                                     title="win rate")))

    mc = ml.get("metric_correlation") or {}
    mm = mc.get("metrics") or []
    if mm and mc.get("matrix"):
        body.append(_fig("metric correlation across samples",
                         svg_heatmap(mm, mm, mc.get("matrix"), fmt="{:.2f}",
                                     title="metric correlation")))

    en = ml.get("ensemble") or {}
    single = en.get("single") or {}
    if single:
        pairs = [(m, single[m]) for m in sorted(single)]
        if "oracle" in en:
            pairs.append(("oracle (best per line)", en["oracle"]))
        if "rover" in en:
            pairs.append(("ROVER majority vote", en["rover"]))
        body.append(_fig("single-model CER vs oracle / ROVER ceilings",
                         hbar(pairs)))

    bc = ml.get("bootstrap_cer") or {}
    rows = []
    for m in sorted(bc):
        for ds in sorted(bc[m] or {}):
            d = bc[m][ds]
            rows.append((f"{m} / {ds}", d.get("point"), d.get("lo"),
                         d.get("hi")))
    if rows:
        body.append(_fig("bootstrap 95% CI on CER-light (model × dataset)",
                         forest(rows)))

    rs = ml.get("rank_stability") or {}
    rrows = []
    for m in sorted(k for k in rs if k != "datasets"):
        for ds in sorted(rs[m] or {}):
            d = rs[m][ds]
            rrows.append((m, ds, d.get("best"), d.get("mode"), d.get("worst")))
    if rrows:
        body.append(_fig("bootstrap rank stability",
                         _tbl(["model", "dataset", "best rank", "modal rank",
                               "worst rank"], rrows)))

    pdd = ml.get("profile_decomposition") or {}
    if pdd:
        rows = [(m, pdd[m]) for m in sorted(pdd)]
        body.append(_fig("normalization-profile decomposition of CER-raw",
                         stacked_hbar(rows,
                                      ["real", "orthographic", "diacritic"])))

    return _panel("model-landscape", "Model landscape", note,
                  "".join(body) or _NA)


def _render_operational(findings: dict) -> str:
    note = ("Cost, latency and reliability: the cost-accuracy Pareto frontier, "
            "latency spread, failure-cause mix, and retry / empty-output rates.")
    op = findings.get("operational") or {}
    if not op:
        return _panel("operational", "Operations", note, _NA)
    body = []

    pa = op.get("pareto") or {}
    if pa:
        ms = sorted(pa)
        pts = [((pa[m] or {}).get("usd_per_1k"), (pa[m] or {}).get("cer_light"))
               for m in ms]
        body.append(_fig("cost – accuracy Pareto (log cost axis)",
                         svg_scatter(pts, x_label="USD / 1k lines",
                                     y_label="CER-light", pareto=True,
                                     log_x=True, labels=ms)))

    lat = op.get("latency") or {}
    rows = []
    for m in sorted(lat):
        d = lat[m]
        p50, p95 = _f(d.get("p50")), _f(d.get("p95"))
        med = min(max(_f(d.get("mean")), p50), p95)
        rows.append((m, {"min": p50, "q1": p50, "median": med, "q3": p95,
                         "max": p95}))
    if rows:
        body.append(_fig("latency p50 → p95 (ms), mean as the median line",
                         box_row(rows)))

    fcs = op.get("failure_causes") or {}
    classes = list(FAILURE_CLASSES)
    rows = []
    for m in sorted(fcs):
        for ds in sorted(fcs[m] or {}):
            rows.append((f"{m} / {ds}", fcs[m][ds]))
    if rows:
        body.append(_fig("failure-cause decomposition (model × dataset)",
                         stacked_hbar(rows, classes)))

    rt = op.get("retries") or {}
    bp = rt.get("by_provider") or {}
    if bp:
        body.append(_fig("lines retried, by provider",
                         hbar([(m, (bp[m] or {}).get("retried"))
                               for m in sorted(bp)])))
    ah = rt.get("attempts_hist") or {}
    if ah:
        body.append(_fig("attempts per line", _tbl(
            ["attempts", "lines"],
            [(k, ah[k]) for k in sorted(ah, key=lambda x: int(x))])))

    er = op.get("empty_rate") or {}
    if er:
        body.append(_fig("empty / near-empty output rate", _tbl(
            ["model", "empty", "n", "rate"],
            [(m, (er[m] or {}).get("empty"), (er[m] or {}).get("n"),
              _pct((er[m] or {}).get("rate"), 1)) for m in sorted(er)])))

    return _panel("operational", "Operations", note, "".join(body) or _NA)


def _render_corpus_health(findings: dict) -> str:
    note = ("Health of the benchmark itself: per-dataset character statistics "
            "and Zipf curves, how model ranks shift across metrics, and any "
            "contamination flags.")
    ch = findings.get("corpus_health") or {}
    if not ch:
        return _panel("corpus-health", "Corpus health", note, _NA)
    body = []

    gs = ch.get("gt_stats") or {}
    zipf = {ds: (gs[ds].get("zipf") or []) for ds in gs}
    zipf = {ds: v for ds, v in zipf.items() if v}
    if zipf:
        body.append(_fig("character-frequency Zipf curve per dataset",
                         binned_curve(zipf, x_label="rank",
                                      y_label="count")))
    if gs:
        rows = [(ds, gs[ds].get("charset_size"),
                 _num(gs[ds].get("mean_line_chars"), 1),
                 _pct(gs[ds].get("diac_density"), 1),
                 gs[ds].get("n_lines")) for ds in sorted(gs)]
        body.append(_fig("ground-truth statistics per dataset",
                         _tbl(["dataset", "charset size", "mean line chars",
                               "diac density", "lines"], rows)))

    churn = ch.get("leaderboard_churn") or {}
    ranks = churn.get("ranks") or {}
    if ranks:
        db = []
        for m in sorted(ranks):
            vs = [v for v in ranks[m].values() if isinstance(v, (int, float))]
            if vs:
                db.append((m, min(vs), max(vs)))
        if db:
            body.append(_fig("rank spread across metrics",
                             dumbbell(db, left_label="best rank",
                                      right_label="worst rank")))
        cols = list(churn.get("metrics") or [])
        if any("cost_adjusted" in (ranks[m] or {}) for m in ranks):
            cols = cols + ["cost_adjusted"]
        body.append(_fig("rank under each metric", _tbl(
            ["model"] + cols,
            [[m] + [(ranks[m] or {}).get(c) for c in cols]
             for m in sorted(ranks)])))

    cont = ch.get("contamination") or {}
    flagged = cont.get("flagged") or []
    if flagged:
        body.append(_fig("contamination-flagged lines", _tbl(
            list(flagged[0].keys()) if isinstance(flagged[0], dict) else ["line"],
            [list(r.values()) if isinstance(r, dict) else [r] for r in flagged])))
    elif not cont.get("available", False):
        body.append('<p class="board-note">No contamination report available '
                    '(run <code>training/contamination.py</code>).</p>')
    else:
        body.append('<p class="board-note">No lines flagged for '
                    'contamination.</p>')

    return _panel("corpus-health", "Corpus health", note, "".join(body) or _NA)


_RENDERERS = {
    "coverage": _render_coverage,
    "error-anatomy": _render_error_anatomy,
    "script-letters": _render_script_letters,
    "diacritics": _render_diacritics,
    "difficulty": _render_difficulty,
    "model-landscape": _render_model_landscape,
    "operational": _render_operational,
    "corpus-health": _render_corpus_health,
}


def render_findings_view(findings: dict) -> str:
    """The whole Findings view: a ``<div class="findings-view">`` wrapping the
    eight sections in ``FINDINGS_SECTIONS`` order. Always well-formed and always
    carrying every ``id='f-<slug>'`` anchor; sections with no data render their
    shell plus a "not available" note."""
    findings = findings or {}
    parts = ['<div class="findings-view">']
    for slug, _label in FINDINGS_SECTIONS:
        parts.append(_RENDERERS[slug](findings))
    parts.append("</div>")
    return "".join(parts)
