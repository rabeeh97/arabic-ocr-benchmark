"""Generate the two inline SVG charts for the research-record page."""
import json

# ---- 1. progress: best official-test CER after each kept step --------------------------------------------------------------------
steps = [  # label, RASAM-1, RASAM-2 (official test, with LM unless noted)
    ("RASAM-1 only + LM (A1)", 5.46, 21.59),
    ("Train on both sets (A3)", 5.35, 10.45),
    ("96 px lines", 4.58, 9.66),
    ("+ TariMa", 4.32, 9.41),
    ("60 epochs", 4.33, 9.13),
    ("+ AraGPT2 rescoring", 4.12, 8.92),
]
W, H, L, R, T, B = 760, 300, 46, 20, 24, 74
xmax = len(steps) - 1
ymax = 24


def X(i):
    return L + (W - L - R) * i / xmax


def Y(v):
    return T + (H - T - B) * (1 - v / ymax)


s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Our official-test CER after each kept step" class="chart">']
for v in range(0, ymax + 1, 4):
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
    s.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v}%</text>')
for key, cls, idx in (("RASAM-2", "s2", 2), ("RASAM-1", "s1", 1)):
    pts = " ".join(f"{X(i):.1f},{Y(st[idx]):.1f}" for i, st in enumerate(steps))
    s.append(f'<polyline points="{pts}" class="line {cls}" fill="none"/>')
    for i, st in enumerate(steps):
        s.append(f'<circle cx="{X(i):.1f}" cy="{Y(st[idx]):.1f}" r="{4.5 if i == xmax else 3.2}" class="dot {cls}"/>')
        dy = -10 if cls == "s2" else 17
        s.append(f'<text x="{X(i):.1f}" y="{Y(st[idx])+dy:.1f}" class="val {cls}" text-anchor="middle">{st[idx]:.2f}</text>')
for i, st in enumerate(steps):
    words = st[0].split(" ")
    half = (len(words) + 1) // 2
    for j, part in enumerate((" ".join(words[:half]), " ".join(words[half:]))):
        if part:
            s.append(f'<text x="{X(i):.1f}" y="{H-B+22+j*14}" class="cat" text-anchor="middle">{part}</text>')
s.append(f'<text x="{X(xmax)-4:.1f}" y="{Y(8.92)-26:.1f}" class="key s2" text-anchor="end">RASAM-2</text>')
s.append(f'<text x="{X(xmax)-4:.1f}" y="{Y(4.12)+34:.1f}" class="key s1" text-anchor="end">RASAM-1</text>')
s.append("</svg>")
progress = "\n".join(s)

# ---- 2. leave-manuscript-out dumbbell (official test lines, LM) -------------------------------------------------------------------
d = json.load(open("results/rasam/lomo_report.json"))["per_manuscript"]
rows = []
for k, v in d.items():
    name = k.replace("BULAC_", "").replace("MS_ARA_", "MS ").replace("ARA_MS_", "MS ")
    rows.append((name, v["test_n"], v["test_insplit_lm_mean"], v["test_lomo_lm"]))
regular = sorted([r for r in rows if r[0] != "MS 1936"], key=lambda r: r[3] - r[2], reverse=True)
W2, rowh, L2, R2, T2 = 760, 26, 92, 70, 30
H2 = T2 + rowh * len(regular) + 34
xm = 18


def X2(v):
    return L2 + (W2 - L2 - R2) * v / xm


s = [f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Per-manuscript CER, seen vs unseen manuscript" class="chart">']
for v in range(0, xm + 1, 3):
    s.append(f'<line x1="{X2(v):.1f}" x2="{X2(v):.1f}" y1="{T2-8}" y2="{H2-28}" class="grid"/>')
    s.append(f'<text x="{X2(v):.1f}" y="{H2-12}" class="tick" text-anchor="middle">{v}%</text>')
for i, (name, n, ins, lomo) in enumerate(regular):
    y = T2 + rowh * i + rowh / 2
    s.append(f'<text x="{L2-10}" y="{y+4:.1f}" class="cat" text-anchor="end">{name}</text>')
    s.append(f'<line x1="{X2(ins):.1f}" x2="{X2(lomo):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="bar"/>')
    s.append(f'<circle cx="{X2(ins):.1f}" cy="{y:.1f}" r="5" class="dot s1"/>')
    s.append(f'<circle cx="{X2(lomo):.1f}" cy="{y:.1f}" r="5" class="dot s2"/>')
    s.append(f'<text x="{X2(max(ins, lomo))+10:.1f}" y="{y+4:.1f}" class="val">+{lomo-ins:.2f}</text>')
s.append(f'<circle cx="{L2+4}" cy="12" r="5" class="dot s1"/><text x="{L2+14}" y="16" class="key">seen manuscript (random line split, 3-seed mean)</text>')
s.append(f'<circle cx="{L2+330}" cy="12" r="5" class="dot s2"/><text x="{L2+340}" y="16" class="key">unseen manuscript (held-out fold)</text>')
s.append("</svg>")
lomo = "\n".join(s)

open("/Users/rabeehjouni/.claude/jobs/abb5a0e7/tmp/chart_progress.svg", "w").write(progress)
open("/Users/rabeehjouni/.claude/jobs/abb5a0e7/tmp/chart_lomo.svg", "w").write(lomo)
print(len(progress), len(lomo), [r[0] for r in regular])
