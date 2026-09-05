"""Coordinate a benchmark sweep run from several Colab accounts at once.

Two or three Colab runners mount the **same** Google Drive and run the **same**
notebook. This module lets them divide the work without a lock service.

Model — advisory leases over a shared Drive directory (``claims/``):

* A unit of work is ``(model, dataset, [i0, i1))`` — a ``BATCH``-aligned slice of
  absolute dataset row indices.
* To take a unit a runner writes a lease file, waits a jittered moment, then
  re-reads. On a collision the deterministic tiebreak — earliest ``claimed_at``,
  then lexicographically smallest ``session_id`` — picks the winner; the loser
  deletes its lease and tries another unit.
* Correctness never depends on a lease being honoured. The per-line result cache
  already stops genuine duplicate spend, and every runner writes its own
  ``results/<run_id>/`` directories, so a lost race just produces one extra
  transcription — which :mod:`scripts.sweep_audit` will surface.
* Recovery: a ``reserved`` lease older than ``ttl_sec`` is abandoned and any
  runner may reclaim it. On start a runner also re-runs its own still-open
  leases first (a fast path for when its id survived a VM restart).

Runner identity is the Colab Google-account email when reachable (stable across
VM restarts), else a uuid persisted at ``/content/.sweep_runner_id``; a fresh
per-session nonce is always appended so two sessions of one account stay
distinguishable in the audit.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

LEASE_TTL_SEC = 30 * 60
_RUNNER_ID_FILE = Path("/content/.sweep_runner_id")
_SAFE = re.compile(r"[^A-Za-z0-9._+@-]")


def _safe(s: str) -> str:
    return _SAFE.sub("-", str(s))


# --------------------------------------------------------------------------- #
# runner identity
# --------------------------------------------------------------------------- #
def _colab_email() -> str | None:
    """The signed-in Colab account's email, via the credentials the notebook
    already obtained to mount Drive. Best-effort — returns None off Colab or if
    the userinfo call fails."""
    try:
        import google.auth
        import google.auth.transport.requests
        import requests

        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        r = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=10,
        )
        if r.ok:
            email = r.json().get("email")
            if email:
                return str(email)
    except Exception:  # noqa: BLE001,S110 - any failure -> fall back to a uuid
        pass
    return None


def _persisted_uuid() -> str:
    try:
        if _RUNNER_ID_FILE.exists():
            v = _RUNNER_ID_FILE.read_text().strip()
            if v:
                return v
    except OSError:
        pass
    v = f"vm-{uuid.uuid4().hex[:12]}"
    try:
        _RUNNER_ID_FILE.write_text(v)
    except OSError:
        pass
    return v


def runner_identity(explicit: str | None = None) -> tuple[str, str]:
    """``(runner_id, session_id)``.

    ``runner_id`` is stable — used for attribution and same-runner resume.
    ``session_id`` is fresh on every call, so duplicate work produced by two
    sessions of the same account is still distinguishable.
    """
    rid = (
        explicit
        or os.environ.get("SWEEP_RUNNER_ID")
        or _colab_email()
        or _persisted_uuid()
    )
    rid = _safe(rid)[:80] or "unknown"
    return rid, uuid.uuid4().hex[:8]


# --------------------------------------------------------------------------- #
# leases
# --------------------------------------------------------------------------- #
@dataclass
class Lease:
    model: str
    dataset: str
    i0: int
    i1: int
    indices: list[int]
    runner_id: str
    session_id: str
    status: str = "reserved"  # "reserved" | "done"
    claimed_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_id: str = ""
    n_scored: int = 0
    path: str = ""  # local: the lease file it was loaded from / written to

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.model, self.dataset, self.i0, self.i1)

    def age(self) -> float:
        return time.time() - self.claimed_at

    def expired(self, ttl: float = LEASE_TTL_SEC) -> bool:
        return self.status == "reserved" and self.age() > ttl

    def overlaps(self, other: Lease) -> bool:
        return (
            self.model == other.model
            and self.dataset == other.dataset
            and self.i0 < other.i1
            and other.i0 < self.i1
        )

    def wins_over(self, other: Lease) -> bool:
        """Deterministic collision tiebreak: earliest claim, then session id."""
        return (self.claimed_at, self.session_id) <= (other.claimed_at, other.session_id)


_LEASE_FIELDS = {f.name for f in fields(Lease)}


class Ledger:
    """A directory of lease files. Everything is best-effort: a torn read is
    skipped, a failed unlink is ignored — the TTL and the result cache are the
    real safety net."""

    def __init__(self, claims_dir: str | Path, ttl_sec: float = LEASE_TTL_SEC):
        self.dir = Path(claims_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = float(ttl_sec)

    # ---- read ------------------------------------------------------------- #
    def _load_all(self) -> list[Lease]:
        out: list[Lease] = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                body = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue  # torn write from another runner — it'll settle
            try:
                lease = Lease(**{k: v for k, v in body.items()
                                 if k in _LEASE_FIELDS and k != "path"})
            except TypeError:
                continue
            lease.path = str(p)
            out.append(lease)
        return out

    def active(self) -> list[Lease]:
        """Leases that still cover their indices: ``done``, or ``reserved`` and
        not past the TTL."""
        return [
            lease for lease in self._load_all()
            if lease.status == "done" or not lease.expired(self.ttl)
        ]

    def covered_indices(self, model: str, dataset: str) -> set[int]:
        s: set[int] = set()
        for lease in self.active():
            if lease.model == model and lease.dataset == dataset:
                s.update(lease.indices)
        return s

    def my_open(self, runner_id: str) -> list[Lease]:
        """This runner's still-``reserved`` leases — work it may have been
        interrupted mid-batch. Re-run these first."""
        return [
            lease for lease in self._load_all()
            if lease.runner_id == runner_id and lease.status == "reserved"
        ]

    def reclaimable(self) -> list[Lease]:
        """``reserved`` leases past the TTL — a crashed runner's abandoned work,
        free for anyone to take over."""
        return [lease for lease in self._load_all() if lease.expired(self.ttl)]

    def stale(self, ttl_sec: float | None = None) -> list[Lease]:
        ttl = self.ttl if ttl_sec is None else float(ttl_sec)
        return [lease for lease in self._load_all() if lease.expired(ttl)]

    # ---- write ---------------------------------------------------------- #
    def _fname(self, lease: Lease) -> str:
        return (f"{_safe(lease.model)}__{_safe(lease.dataset)}__"
                f"{lease.i0:06d}-{lease.i1:06d}__{lease.session_id}.json")

    def _write(self, lease: Lease) -> Lease:
        lease.updated_at = time.time()
        body = {k: v for k, v in asdict(lease).items() if k != "path"}
        p = self.dir / self._fname(lease)
        p.write_text(json.dumps(body, ensure_ascii=False))
        lease.path = str(p)
        return lease

    def claim(self, model: str, dataset: str, indices: list[int],
              runner_id: str, session_id: str) -> Lease | None:
        if not indices:
            return None
        idx = sorted(indices)
        return self._write(Lease(
            model=model, dataset=dataset, i0=idx[0], i1=idx[-1] + 1,
            indices=idx, runner_id=runner_id, session_id=session_id,
        ))

    def confirm(self, lease: Lease, *, jitter: tuple[float, float] = (2.0, 5.0)) -> bool:
        """Pause a jittered moment, re-read the directory, and return True iff
        this lease still beats every overlapping ``reserved`` lease."""
        lo, hi = jitter
        if hi > 0:
            time.sleep(random.uniform(lo, hi))
        for other in self._load_all():
            if other.path == lease.path or other.status != "reserved":
                continue
            if other.expired(self.ttl):
                continue
            if lease.overlaps(other) and not lease.wins_over(other):
                return False
        return True

    def adopt(self, lease: Lease, runner_id: str, session_id: str) -> Lease | None:
        """Take an expired (or one's own) lease over under my identity, as a
        fresh file. Call :meth:`confirm` on the result before doing the work."""
        self.release(lease)
        return self.claim(lease.model, lease.dataset, lease.indices,
                          runner_id, session_id)

    def complete(self, lease: Lease, run_id: str, n_scored: int) -> Lease:
        lease.status = "done"
        lease.run_id = run_id
        lease.n_scored = int(n_scored)
        return self._write(lease)

    def release(self, lease: Lease) -> None:
        try:
            Path(lease.path).unlink()
        except OSError:
            pass

    def prune_done(self, keep_sec: float = 7 * 24 * 3600) -> int:
        """Delete ``done`` lease files older than ``keep_sec`` (housekeeping so
        the directory doesn't grow without bound). Returns how many were removed."""
        n = 0
        for lease in self._load_all():
            if lease.status == "done" and (time.time() - lease.updated_at) > keep_sec:
                self.release(lease)
                n += 1
        return n
