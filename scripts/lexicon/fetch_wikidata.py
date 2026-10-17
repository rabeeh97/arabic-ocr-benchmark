"""Fetch Arabic labels and aliases of name-bearing Wikidata classes (CC0) from the QLever endpoint, paged.

    python -I scripts/lexicon/fetch_wikidata.py --out /Volumes/PortableSSD/data/lexicon/wikidata

Writes <class>.tsv.gz with rows: qid \t kind(label|alias) \t text. Pages of 100k rows with retries, because the
link from this machine is slow and long single responses get cut.
"""
from __future__ import annotations

import argparse
import gzip
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT = "https://qlever.dev/api/wikidata"
PREFIX = ("PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> "
          "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> ")
CLASSES = {
    "person": "wdt:P31 wd:Q5",
    "given_name": "wdt:P31/wdt:P279* wd:Q202444",
    "family_name": "wdt:P31/wdt:P279* wd:Q101352",
    "country": "wdt:P31/wdt:P279* wd:Q6256",
    "admin_area": "wdt:P31/wdt:P279* wd:Q56061",
    "settlement": "wdt:P31/wdt:P279* wd:Q486972",
    "organisation": "wdt:P31/wdt:P279* wd:Q43229",
}
PAGE = 100_000


def query(q: str, tries: int = 6) -> str:
    data = urllib.parse.urlencode({"query": q}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers={"Accept": "text/tab-separated-values"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001 - network: retry anything
            print(f"  retry {i + 1}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError("query failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, pat in CLASSES.items():
        if a.only and name not in a.only:
            continue
        dest = a.out / f"{name}.tsv.gz"
        if dest.exists() and dest.stat().st_size > 0 and (a.out / f"{name}.done").exists():
            continue
        n = 0
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            for kind, prop in (("label", "rdfs:label"), ("alias", "skos:altLabel")):
                off = 0
                while True:
                    q = (PREFIX + f"SELECT ?x ?l WHERE {{ ?x {pat} . ?x {prop} ?l . FILTER(LANG(?l)=\"ar\") }} "
                         f"LIMIT {PAGE} OFFSET {off}")
                    lines = query(q).splitlines()[1:]
                    for ln in lines:
                        x, _, lab = ln.partition("\t")
                        qid = x.rsplit("/", 1)[-1].rstrip(">")
                        if lab.endswith('"@ar'):
                            lab = lab[1:-4]
                        lab = lab.replace("\\t", " ").replace("\\n", " ").replace('\\"', '"')
                        fh.write(f"{qid}\t{kind}\t{lab}\n")
                    n += len(lines)
                    print(f"{name} {kind} offset {off}: +{len(lines)} (total {n})", flush=True)
                    if len(lines) < PAGE:
                        break
                    off += PAGE
        (a.out / f"{name}.done").write_text(datetime.now(timezone.utc).isoformat(timespec="minutes") + f"\t{n}\n")


if __name__ == "__main__":
    main()
