"""Fingerprint of a line pickle (roles, labels, image bytes) to check that two machines hold the same data.

    python pickle_fingerprint.py FILE.pkl
"""
import hashlib
import pickle
import sys
from collections import Counter

rows = pickle.load(open(sys.argv[1], "rb"))["rows"]
h = hashlib.sha256()
for r in rows:
    h.update(r["role"].encode() + r["label"].encode("utf-8") + r["img"].tobytes())
print(Counter(r["role"] for r in rows), h.hexdigest()[:16])
