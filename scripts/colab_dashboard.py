"""Local HTML dashboard for everything that runs on Colab: account, budget, live sessions, running and finished jobs, curves, hardware comparison.

    python3 scripts/colab_dashboard.py                 write the page once
    python3 scripts/colab_dashboard.py --loop 30       keep rewriting it every 30 s (run detached); the page reloads itself every 30 s

Output: ~/colab_staging/dashboard/index.html (open it with `open ~/colab_staging/dashboard/index.html`); a copy goes next to the results on the SSD when it is mounted.
Sources (all under ~/colab_staging unless noted): jobs/*/job_record.json (finished jobs), jobs/*/job_progress.log (the full training curve captured on every poll),
logs/*.log (supervisor logs, also parsed for jobs started before job_progress.log existed), budget.json, history.json (recovered numbers from runs whose weights were lost),
the live `colab usage` and `colab sessions`, and the Mac-side queue log on the SSD. A source that cannot be read is shown as unavailable; the page never fails because of one.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

HOME = Path.home()
ST = HOME / "colab_staging"
OUT = ST / "dashboard"
REPO = Path("/Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike")
QUEUE_LOG = Path("/Volumes/PortableSSD/models/rasam-replication/crnn-runs/queue_master.log")
EPOCH = re.compile(r"epoch (\d+)/(\d+)\s+loss ([\d.]+)\s+(\d+)s(?:\s+heldout CER (\{.*?\}))?")
CACHE = ST / "dashboard_last_live.json"


def run(cmd: list[str], timeout: int = 60) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def live() -> dict:
    usage = run(["colab", "usage"])
    sess = run(["colab", "sessions"])
    d = {"t": time.strftime("%H:%M:%S"), "ok": False}
    m = re.search(r"Current balance:\s*([\d.]+)", usage)
    if m:
        d["ok"] = True
        d["balance"] = float(m.group(1))
        d["rate"] = float((re.search(r"Usage rate:\s*([\d.]+)", usage) or [0, 0])[1])
        d["active"] = int((re.search(r"Active assignments:\s*(\d+)", usage) or [0, 0])[1])
        d["sessions"] = [l.strip() for l in sess.splitlines() if l.startswith("[") and "Hardware" in l]
        CACHE.write_text(json.dumps(d))
    elif CACHE.exists():
        d = json.loads(CACHE.read_text())
        d["stale"] = True
    return d


def parse_epochs(text: str) -> list[dict]:
    seen: dict[tuple, dict] = {}
    run_i = 0
    last_e = 0
    for line in text.splitlines():
        m = EPOCH.search(line)
        if not m:
            continue
        e, n, loss, sec = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
        if e < last_e:  # a new training run inside the same job (epoch counter restarted)
            run_i += 1
        last_e = e
        cer = ast.literal_eval(m.group(5)) if m.group(5) else None
        seen[(run_i, e)] = {"run": run_i, "epoch": e, "of": n, "loss": loss, "sec": sec, "cer": cer}
    return sorted(seen.values(), key=lambda r: (r["run"], r["epoch"]))


def jobs() -> list[dict]:
    out = []
    for d in sorted((ST / "jobs").glob("*")):
        if not d.is_dir():
            continue
        j: dict = {"name": d.name, "state": "running", "units": None, "minutes": None, "reason": "", "gpu": "?", "weights": []}
        rec = d / "job_record.json"
        if rec.exists():
            r = json.loads(rec.read_text())
            j.update(state="finished", units=r.get("units_used"), minutes=r.get("minutes"), reason=r.get("reason", ""), gpu=r.get("spec", {}).get("gpu", "?"))
        else:
            m = re.search(r"job-.*?-(t4|l4|a100|g4|h100|cpu)-\d+$", d.name)
            j["gpu"] = m.group(1).upper() if m else "?"
        text = ""
        prog = d / "job_progress.log"
        if prog.exists():
            text = prog.read_text(errors="replace")
        if not text:  # older jobs: the supervisor log has the epoch lines of each poll
            for lg in (ST / "logs").glob("*.log"):
                body = lg.read_text(errors="replace")
                if d.name.split("-", 1)[1].rsplit("-", 1)[0].replace("-", "") in lg.stem.replace("_", "").replace("-", "") or lg.stem.replace("_", "-") in d.name:
                    text += body
        j["epochs"] = parse_epochs(text)
        j["weights"] = [f"{p.relative_to(d)} ({p.stat().st_size / 1e6:.1f} MB)" for p in d.rglob("*.pt")]
        poll = [l for l in text.splitlines() if l.startswith("[")]
        j["last_poll"] = poll[-1][:200] if poll else ""
        out.append(j)
    return out


def svg_chart(series: dict[str, list[tuple[int, float]]], w: int = 300, h: int = 110) -> str:
    pts = [p for s in series.values() for p in s]
    if len(pts) < 2:
        return "<div class=dim>no curve yet</div>"
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), 0.0, max(ys)
    sx = lambda x: 28 + (x - x0) / max(1, x1 - x0) * (w - 36)  # noqa: E731
    sy = lambda y: h - 18 - (y - y0) / max(1e-9, y1 - y0) * (h - 28)  # noqa: E731
    colors = ["var(--c1)", "var(--c2)", "var(--c3)"]
    body = f'<line x1="28" y1="{h - 18}" x2="{w - 6}" y2="{h - 18}" stroke="var(--grid)"/><line x1="28" y1="10" x2="28" y2="{h - 18}" stroke="var(--grid)"/>'
    body += f'<text x="2" y="14" class=ax>{y1 * 100:.0f}%</text><text x="2" y="{h - 20}" class=ax>0</text><text x="{w - 40}" y="{h - 4}" class=ax>epoch {x1}</text>'
    for i, (name, s) in enumerate(series.items()):
        if len(s) >= 1:
            d = " ".join(f"{'M' if k == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for k, (x, y) in enumerate(s))
            body += f'<path d="{d}" fill="none" stroke="{colors[i % 3]}" stroke-width="2"/><text x="{w - 70}" y="{14 + 12 * i}" class=ax fill="{colors[i % 3]}" style="fill:{colors[i % 3]}">{html.escape(name)} {s[-1][1] * 100:.1f}%</text>'
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'


def job_card(j: dict) -> str:
    ep = j["epochs"]
    series: dict[str, list] = {}
    for r in ep:
        if r["cer"]:
            for k, v in r["cer"].items():
                series.setdefault(f"{k} run{r['run'] + 1}" if max(x["run"] for x in ep) else k, []).append((r["epoch"], v))
    last = ep[-1] if ep else None
    secs = [r["sec"] for r in ep if r["epoch"] > 1]
    stat = f"epoch {last['epoch']}/{last['of']}, loss {last['loss']}, {sum(secs) / len(secs):.0f} s/epoch" if last and secs else ("starting up" if j["state"] == "running" else "no epoch lines recorded")
    badge = f'<span class="b {j["state"]}">{j["state"]}</span>'
    w = ", ".join(j["weights"]) if j["weights"] else "none saved yet"
    return (f'<div class=card><h3>{html.escape(j["name"])} {badge}</h3>'
            f'<div class=kv><span>GPU</span><b>{html.escape(str(j["gpu"]))}</b><span>units</span><b>{j["units"] if j["units"] is not None else "-"}</b>'
            f'<span>minutes</span><b>{j["minutes"] if j["minutes"] is not None else "-"}</b></div>'
            f'<div class=dim>{html.escape(stat)}</div>{svg_chart(series)}'
            f'<div class=dim>weights: {html.escape(w)}</div><div class=dim>{html.escape(j["reason"])}</div>'
            f'<div class=dim mono>{html.escape(j["last_poll"][:150])}</div></div>')


def history_cards(h: dict) -> str:
    s = ""
    for r in h.get("runs", []):
        series = {"r1": [(p[0], p[1]) for p in r["points"]], "r2": [(p[0], p[2]) for p in r["points"]]}
        s += (f'<div class="card warn"><h3>{html.escape(r["id"])} <span class="b lost">recovered numbers only</span></h3>'
              f'<div class=kv><span>GPU</span><b>{r["gpu"]}</b><span>epoch s</span><b>{r.get("epoch_s", "-")}</b><span>units</span><b>~{r.get("units_used_approx", "-")}</b></div>'
              f'<div class=dim>{html.escape(r["config"])}</div>{svg_chart(series)}<div class=dim>{html.escape(r["note"])}</div></div>')
    return s


def hardware_table(js: list[dict], h: dict) -> str:
    rate = h.get("measured", {}).get("gpu_units_per_hour", {})
    rows = []
    for g in ("T4", "A100", "CPU"):
        secs = [r["sec"] for j in js if str(j["gpu"]).upper() == g for r in j["epochs"] if r["epoch"] > 1]
        secs += [r.get("epoch_s") for r in h.get("runs", []) if r["gpu"] == g and r.get("epoch_s")]
        if g in rate:
            es = f"{sum(secs) / len(secs):.0f}" if secs else "-"
            per = f"{(sum(secs) / len(secs)) / 3600 * rate[g]:.3f}" if secs else "-"
            rows.append(f"<tr><td>{g}</td><td>{rate[g]}</td><td>{es}</td><td>{per}</td></tr>")
    rows.append("<tr><td>Mac CPU (half width, shared)</td><td>-</td><td>~730 (A3), ~546 (RASAM-1 only)</td><td>-</td></tr>")
    return "<table><tr><th>hardware</th><th>units / hour</th><th>s / epoch (A3)</th><th>units / epoch</th></tr>" + "".join(rows) + "</table>"


def mac_queue() -> str:
    if not QUEUE_LOG.exists():
        return "<div class=dim>SSD not mounted or no queue log</div>"
    lines = [l for l in QUEUE_LOG.read_text(errors="replace").splitlines() if re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) ", l)]
    a3 = ""
    lg = QUEUE_LOG.parent / "A3_w05_s0.log"
    if lg.exists():
        e = [l for l in lg.read_text(errors="replace").splitlines() if l.startswith("epoch")]
        a3 = f"<div class=dim>Mac A3 (CPU): {html.escape(e[-1][:120]) if e else 'no epochs yet'}</div>"
    return a3 + "<pre>" + html.escape("\n".join(lines[-8:])) + "</pre>"


SSD = Path("/Volumes/PortableSSD")
RUNS = SSD / "models/rasam-replication/crnn-runs"
HF = SSD / "huggingface/hub"
PROC_PATTERNS = [  # (label, regex on the command line)
    ("CPU training", r"xai_train_aligner\.py"), ("LM decode", r"rasam_lm_decode\.py"), ("test scoring", r"rasam_crnn_eval\.py"),
    ("Mac queue master", r"queue_master\.sh"), ("Colab supervisor", r"colab_job\.py"), ("downloads track", r"net_track\.sh"),
    ("HF download", r"hf download|snapshot_download"), ("curl download", r"curl .*(zenodo|bulac|huggingface)"), ("page fetch", r"rasam_fetch_pages"),
]


def dir_size(p: Path) -> float:
    total = 0
    for d, _, fs in __import__("os").walk(p):
        for f in fs:
            try:
                total += (Path(d) / f).stat().st_size
            except OSError:
                pass
    return total / 1e9


def processes() -> list[dict]:
    out = run(["ps", "-axo", "pid,etime,%cpu,%mem,command"], 10).splitlines()[1:]
    rows = []
    for line in out:
        parts = line.split(None, 4)
        if len(parts) < 5 or "colab_dashboard" in parts[4] or parts[4].startswith(("grep", "/bin/zsh -c source")):
            continue
        for label, rx in PROC_PATTERNS:
            if re.search(rx, parts[4]) and not parts[4].startswith("caffeinate"):
                m = re.search(r"--out \S*/([\w.\-]+)", parts[4])
                rows.append({"label": label + (f" ({m.group(1)})" if m else ""), "pid": parts[0], "elapsed": parts[1], "cpu": parts[2], "mem": parts[3]})
                break
    return rows


def system_html() -> str:
    import os

    la = os.getloadavg()
    vm = run(["vm_stat"], 10)
    pg = 16384
    free = sum(int(x) for x in re.findall(r"Pages (?:free|speculative):\s+(\d+)", vm)) * pg / 1e9
    disks = []
    for name, p in (("internal disk", Path.home()), ("SSD", SSD)):
        try:
            u = shutil.disk_usage(p)
            disks.append(f"<div class=card><span class=dim>{name} free</span><b>{u.free / 1e9:.0f} GB</b><div class=bar><i style='width:{u.used / u.total * 100:.0f}%'></i></div></div>")
        except OSError:
            disks.append(f"<div class=card><span class=dim>{name}</span><b>not mounted</b></div>")
    return (f"<div class=top><div class=card><span class=dim>load (1 / 5 / 15 min)</span><b>{la[0]:.1f} / {la[1]:.1f} / {la[2]:.1f}</b></div>"
            f"<div class=card><span class=dim>free memory (8.6 GB machine)</span><b>{free:.1f} GB</b></div>{''.join(disks)}</div>")


def cpu_experiments_html() -> str:
    if not RUNS.exists():
        return "<div class=dim>SSD not mounted: CPU runs unavailable</div>"
    rows = []
    for lg in sorted(RUNS.glob("*.log")):
        if lg.name in ("queue.log", "queue_master.log") or "corrupt" in lg.name:
            continue
        text = lg.read_text(errors="replace")
        eps = [l for l in text.splitlines() if l.startswith("epoch")]
        cers = re.findall(r"heldout CER (\{.*?\})", text)
        done = bool(re.search(r"^done", text, re.M))
        state = "finished" if done else ("running" if any(lg.stem in p["label"] for p in processes()) else "stopped")
        last = eps[-1][:60] if eps else "no epochs"
        rows.append(f"<tr><td>{lg.stem}</td><td><span class='b {state}'>{state}</span></td><td>{html.escape(last)}</td><td>{html.escape(cers[-1] if cers else '-')}</td></tr>")
    res = []
    for d, kind in ((REPO / "results/rasam/crnn", "greedy"), (REPO / "results/rasam/lm", "with LM")):
        for f in sorted(d.glob("*__test.json")) if d.exists() else []:
            try:
                j = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            g = j.get("greedy_micro_cer", {}).get("raw")
            res.append(f"<tr><td>{html.escape(f.stem.replace('__test', ''))}</td><td>{kind}</td><td>{j.get('n')}</td><td>{j['micro_cer']['raw'] * 100:.2f} %</td>"
                       f"<td>{f'{g * 100:.2f} %' if g is not None else '-'}</td></tr>")
    steps = ""
    if QUEUE_LOG.exists():
        lines = [l for l in QUEUE_LOG.read_text(errors="replace").splitlines() if "STEP" in l or "QUEUE DONE" in l or "rc=" in l]
        steps = "<pre>" + html.escape("\n".join(lines[-8:])) + "</pre>"
    return ("<h3>Training runs (Mac CPU)</h3><table><tr><th>run</th><th>state</th><th>last epoch</th><th>last held-out CER</th></tr>" + "".join(rows) + "</table>"
            "<h3 style='margin-top:12px'>Mac queue steps</h3>" + steps +
            "<h3>Test-set results so far (raw CER, published RASAM test splits)</h3><table><tr><th>model / test set</th><th>decoding</th><th>n</th><th>CER</th><th>greedy CER</th></tr>"
            + "".join(res) + "</table>")


def downloads_html() -> str:
    paused = (ST / "PAUSE_DOWNLOADS").exists()
    active = [p for p in processes() if p["label"].split(" (")[0] in ("HF download", "curl download", "page fetch", "downloads track")]
    inv = []
    targets = [("RASAM-1", HF / "datasets--calfa-ai--RASAM-1", 2.6), ("RASAM-2", HF / "datasets--calfa-ai--RASAM-2", 0.52), ("TariMa", HF / "datasets--calfa-ai--tarima", 0.23),
               ("Iskandar", HF / "datasets--calfa-ai--iskandar", 1.0), ("Baybars", HF / "datasets--calfa-ai--baybars", 1.83), ("AMIDDA", HF / "datasets--calfa-ai--AMIDDA", 9.45)]
    for name, p, want in targets:
        if not SSD.exists():
            break
        have = dir_size(p) if p.exists() else 0.0
        pct = min(100, have / want * 100)
        state = "complete" if pct >= 97 else ("paused" if paused and name == "AMIDDA" else "partial")
        inv.append(f"<tr><td>{name}</td><td>{have:.2f} / {want} GB</td><td><div class=bar style='width:120px'><i style='width:{pct:.0f}%'></i></div></td><td>{state}</td></tr>")
    if SSD.exists():
        z = SSD / "data/arams_28k_htr/AraMS-28k-HTR.zip"
        log = SSD / "data/fetch_arams.log"
        ver = log.exists() and "VERIFIED" in log.read_text(errors="replace")
        inv.append(f"<tr><td>AraMS-28k-HTR zip</td><td>{(z.stat().st_size / 1e9 if z.exists() else 0):.2f} / 2.42 GB</td><td></td><td>{'complete, md5 verified' if ver else 'not verified'}</td></tr>")
        pages = SSD / "data/rasam-pages/images"
        n = len(list(pages.glob("*.jpg"))) if pages.exists() else 0
        inv.append(f"<tr><td>RASAM page images (BULAC)</td><td>{n} / 547 pages</td><td></td><td>{'complete' if n >= 547 else 'partial'}</td></tr>")
    act = "".join(f"<li>{html.escape(p['label'])}, pid {p['pid']}, running {p['elapsed']}</li>" for p in active) or "<li class=dim>no download running</li>"
    return (f"<div class=card style='margin-bottom:10px'><b>{'PAUSED' if paused else 'allowed'}</b> <span class=dim>downloads track "
            f"{'(marker ~/colab_staging/PAUSE_DOWNLOADS present: remove it to resume)' if paused else ''}</span><ul>{act}</ul></div>"
            "<table><tr><th>dataset</th><th>on the SSD</th><th></th><th>state</th></tr>" + "".join(inv) + "</table>")


def plan_html() -> str:
    f = ST / "plan.json"
    if not f.exists():
        return "<div class=dim>no plan file</div>"
    p = json.loads(f.read_text())
    groups: dict[str, list] = {}
    for it in p["items"]:
        key = it["status"].split(" ")[0]
        groups.setdefault(key, []).append(it)
    order = ["running", "planned", "blocked", "paused", "done"]
    s = f"<div class=dim>plan file updated {html.escape(p.get('updated', '?'))}</div>"
    for k in order + [k for k in groups if k not in order]:
        if k not in groups:
            continue
        s += f"<h3 style='margin-top:12px'>{k.capitalize()} ({len(groups[k])})</h3><table><tr><th>id</th><th>where</th><th>what</th><th>units</th><th>blocker / note</th></tr>"
        for it in groups[k]:
            s += (f"<tr><td>{html.escape(it['id'])}</td><td>{html.escape(it['where'])}</td><td>{html.escape(it['title'])}</td>"
                  f"<td>{it.get('est_units', '-')}</td><td>{html.escape(it.get('blocker') or it.get('note') or '')}</td></tr>")
        s += "</table>"
    return s


def leaderboard_html() -> str:
    """Published RASAM numbers next to ours, per test set, best first. Ours are read from results/rasam/{crnn,lm}/*__test.json (official HF test splits)."""
    f = ST / "leaderboard.json"
    lb = json.loads(f.read_text()) if f.exists() else {"published": [], "targets": {}}
    rows: dict[str, list] = {"RASAM-1": [], "RASAM-2": []}
    for p in lb.get("published", []):
        rows.setdefault(p["test"], []).append((p["cer"], "published", p["system"], p["source"], p["comparable"]))
    for d, kind in ((REPO / "results/rasam/crnn", "greedy"), (REPO / "results/rasam/lm", "with char LM")):
        for jf in sorted(d.glob("*__test.json")) if d.exists() else []:
            try:
                j = json.loads(jf.read_text())
            except Exception:  # noqa: BLE001
                continue
            test = {"r1": "RASAM-1", "r2": "RASAM-2"}.get(j.get("dataset"))
            if not test or "smoke" in jf.stem:
                continue
            rows[test].append((j["micro_cer"]["raw"] * 100, "ours", f"{jf.stem.replace('__test', '')} ({kind})", "this project", f"official test split, n={j.get('n')}, micro CER raw"))
    s = f"<div class=dim>{html.escape(lb.get('note', ''))}</div>"
    for test, items in rows.items():
        if not items:
            continue
        tgt = lb.get("targets", {}).get(test)
        s += f"<h3 style='margin-top:14px'>{test} test{f' · target to beat: {tgt} %' if tgt else ''}</h3><table><tr><th>#</th><th>CER</th><th>who</th><th>system</th><th>source</th><th>comparability</th></tr>"
        for i, (cer, who, sysname, src, comp) in enumerate(sorted(items)):
            badge = "<span class='b finished'>ours</span>" if who == "ours" else "<span class='b'>published</span>"
            beat = " ★" if who == "ours" and tgt and cer < tgt else ""
            s += f"<tr><td>{i + 1}</td><td><b>{cer:.2f} %</b>{beat}</td><td>{badge}</td><td>{html.escape(sysname)}</td><td>{html.escape(src)}</td><td class=dim>{html.escape(comp)}</td></tr>"
        s += "</table>"
    return s


def build() -> str:
    lv = live()
    budget = json.loads((ST / "budget.json").read_text()) if (ST / "budget.json").exists() else {"baseline_balance": None, "budget_units": 10}
    hist = json.loads((ST / "history.json").read_text()) if (ST / "history.json").exists() else {}
    js = jobs()
    spent = (budget["baseline_balance"] - lv["balance"]) if lv.get("balance") is not None and budget["baseline_balance"] else None
    frac = min(1.0, max(0.0, spent / budget["budget_units"])) if spent is not None else 0
    running = [j for j in js if j["state"] == "running"]
    done = [j for j in js if j["state"] == "finished"]
    stale = " (stale: last good reading)" if lv.get("stale") else ""
    sess = "".join(f"<li>{html.escape(s)}</li>" for s in lv.get("sessions", [])) or "<li class=dim>none</li>"
    ev = ""
    for lg in sorted((ST / "logs").glob("*.log"))[-4:]:
        t = [l for l in lg.read_text(errors="replace").splitlines() if not l.startswith(("│", "╭", "╰"))][-4:]
        ev += f"<div class=dim><b>{html.escape(lg.name)}</b></div><pre>{html.escape(chr(10).join(x[:200] for x in t))}</pre>"
    css = """
:root{--bg:#fbfaf7;--fg:#1d1c1a;--dim:#6b675f;--card:#fff;--line:#e4e0d6;--c1:#1f6feb;--c2:#d4540f;--c3:#2a9d55;--grid:#d8d4c8;--ok:#1a7f4b;--warn:#b45309;--bad:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#161513;--fg:#ece9e2;--dim:#9a958a;--card:#1f1e1b;--line:#33312c;--c1:#6ea8ff;--c2:#ff9a5c;--c3:#5fd08a;--grid:#3a3832;--ok:#4cc38a;--warn:#f0a050;--bad:#ff7b72}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px 20px;max-width:1200px;margin:auto}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:13px;margin:0 0 6px;word-break:break-all}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card.warn{border-color:var(--warn)}.top{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.top .card b{font-size:22px;display:block}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--c1)}
.dim{color:var(--dim);font-size:12px}.mono{font-family:ui-monospace,monospace;word-break:break-all}.kv{display:grid;grid-template-columns:auto auto auto auto auto auto;gap:2px 8px;font-size:12px;margin-bottom:4px}.kv span{color:var(--dim)}
.b{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);font-weight:600}.b.running{color:var(--c1)}.b.finished{color:var(--ok)}.b.lost{color:var(--warn)}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid var(--line);padding:4px 8px;text-align:left}pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px;font-size:11px;overflow:auto;margin:2px 0 8px}
svg text.ax{font-size:9px;fill:var(--dim)}"""
    procs = processes()
    ptable = "".join(f"<tr><td>{html.escape(p['label'])}</td><td>{p['pid']}</td><td>{p['elapsed']}</td><td>{p['cpu']}</td><td>{p['mem']}</td></tr>" for p in procs) \
        or "<tr><td colspan=5 class=dim>nothing of ours is running locally</td></tr>"
    n_local = sum(1 for p in procs if p["label"].startswith(("CPU training", "LM decode", "test scoring")))
    n_dl = sum(1 for p in procs if p["label"].startswith(("HF download", "curl download", "page fetch")))
    tabs_css = """nav{display:flex;gap:6px;margin:12px 0 4px;border-bottom:1px solid var(--line)}nav a{padding:6px 12px;border:1px solid var(--line);border-bottom:none;border-radius:6px 6px 0 0;
color:var(--dim);text-decoration:none;font-weight:600}nav a.on{color:var(--fg);background:var(--card)}section{display:none}section.on{display:block}"""
    script = """<script>
function show(t){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===t));document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.hash==='#'+t));}
window.addEventListener('hashchange',()=>show((location.hash||'#colab').slice(1)));show((location.hash||'#colab').slice(1));
setTimeout(()=>location.reload(),30000);</script>"""
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Experiment dashboard</title><style>{css}{tabs_css}</style></head><body>
<h1>Experiment dashboard</h1><div class=dim>updated {time.strftime('%a %d %b %H:%M:%S')} · reloads every 30 s (keeps the tab) · Colab reading at {lv.get('t', '?')}{stale}</div>
<nav><a href="#colab">Colab ({lv.get('active', '?')} live)</a><a href="#local">Local ({n_local} CPU jobs, {n_dl} downloads)</a><a href="#leaderboard">RASAM leaderboard</a><a href="#plan">Plan, blocked and paused</a></nav>
<section id=colab>
<div class=top style="margin-top:10px">
<div class=card><span class=dim>compute-unit balance</span><b>{lv.get('balance', '-')}</b></div>
<div class=card><span class=dim>spent of the {budget['budget_units']}-unit budget</span><b>{f'{spent:.2f}' if spent is not None else '-'}</b><div class=bar><i style="width:{frac * 100:.0f}%"></i></div></div>
<div class=card><span class=dim>usage rate now</span><b>{lv.get('rate', '-')}/h</b></div>
<div class=card><span class=dim>active sessions</span><b>{lv.get('active', '-')}</b><ul style="margin:4px 0 0 16px;padding:0;font-size:12px">{sess}</ul></div></div>
<h2>Running now ({len(running)})</h2><div class=grid>{''.join(job_card(j) for j in running) or '<div class=dim>nothing running on Colab</div>'}</div>
<h2>Finished ({len(done)})</h2><div class=grid>{''.join(job_card(j) for j in done) or '<div class=dim>none yet with the new runner</div>'}</div>
<h2>Earlier runs (numbers recovered, weights lost)</h2><div class=grid>{history_cards(hist)}</div>
<h2>Hardware comparison (measured)</h2>{hardware_table(js, hist)}
<h2>Recent Colab supervisor logs</h2>{ev}
</section>
<section id=local>
<h2>Machine</h2>{system_html()}
<h2>Our processes running on this Mac</h2><table><tr><th>what</th><th>pid</th><th>running for</th><th>% CPU</th><th>% memory</th></tr>{ptable}</table>
<h2>CPU experiments</h2>{cpu_experiments_html()}
<h2>Downloads and datasets</h2>{downloads_html()}
</section>
<section id=leaderboard>
<h2>RASAM leaderboard: published vs ours</h2>{leaderboard_html()}
</section>
<section id=plan>
<h2>Plan, blocked and paused</h2>{plan_html()}
</section>
{script}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            page = build()
            (OUT / "index.html").write_text(page)
            if REPO.exists():
                try:
                    (REPO / "results/rasam").mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(OUT / "index.html", REPO / "results/rasam/colab_dashboard.html")
                except OSError:
                    pass
        except Exception as e:  # noqa: BLE001  never die: show the error on the next page
            (OUT / "index.html").write_text(f"<pre>dashboard error: {html.escape(repr(e))}</pre><meta http-equiv=refresh content=30>")
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
