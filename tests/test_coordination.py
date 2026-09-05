"""Lease ledger for the multi-runner Colab sweep: claim, collide, expire, resume."""

from __future__ import annotations

import time

from arabic_ocr_bench.coordination import Ledger, Lease, runner_identity


def test_runner_identity_is_stable_id_plus_fresh_session(monkeypatch):
    monkeypatch.setattr("arabic_ocr_bench.coordination._colab_email", lambda: None)
    monkeypatch.setenv("SWEEP_RUNNER_ID", "alice@example.com")
    a_id, a_sess = runner_identity()
    b_id, b_sess = runner_identity()
    assert a_id == b_id == "alice@example.com"   # stable, for attribution / resume
    assert a_sess != b_sess                       # fresh every call


def test_claim_then_covered_then_complete(tmp_path):
    led = Ledger(tmp_path / "claims")
    lease = led.claim("qwen2-vl-2b", "kitab_khatt", [0, 1, 2, 3, 4], "alice", "s1")
    assert lease is not None
    assert led.covered_indices("qwen2-vl-2b", "kitab_khatt") == {0, 1, 2, 3, 4}
    assert led.covered_indices("qwen2-vl-2b", "kitab_muharaf") == set()

    led.complete(lease, "20260101T000000Z", 5)
    # a done lease still covers its indices (so nobody re-claims them)
    assert led.covered_indices("qwen2-vl-2b", "kitab_khatt") == {0, 1, 2, 3, 4}
    # ...and a fresh ledger over the same dir sees it
    assert Ledger(tmp_path / "claims").covered_indices("qwen2-vl-2b", "kitab_khatt") == {0, 1, 2, 3, 4}


def test_claim_returns_none_for_empty_range(tmp_path):
    assert Ledger(tmp_path / "c").claim("m", "d", [], "alice", "s1") is None


def test_collision_is_resolved_deterministically(tmp_path):
    led = Ledger(tmp_path / "claims")
    early = led.claim("m", "d", [0, 1, 2], "alice", "aaaa")
    time.sleep(0.01)
    late = led.claim("m", "d", [1, 2, 3], "bob", "zzzz")   # overlaps [1,2]

    # earliest claimed_at wins; alice keeps it, bob must yield
    assert led.confirm(early, jitter=(0, 0)) is True
    assert led.confirm(late, jitter=(0, 0)) is False


def test_expired_lease_is_reclaimable_by_anyone(tmp_path):
    led = Ledger(tmp_path / "claims", ttl_sec=0.05)
    stale = led.claim("m", "d", [0, 1, 2], "alice", "s1")
    time.sleep(0.08)

    assert led.reclaimable() and led.reclaimable()[0].runner_id == "alice"
    # an expired lease no longer holds its indices
    assert led.covered_indices("m", "d") == set()

    taken = led.adopt(stale, "bob", "s2")
    assert taken.runner_id == "bob"
    assert led.confirm(taken, jitter=(0, 0)) is True
    assert led.reclaimable() == []          # alice's file is gone
    assert led.covered_indices("m", "d") == {0, 1, 2}


def test_my_open_finds_this_runners_unfinished_work(tmp_path):
    led = Ledger(tmp_path / "claims")
    mine = led.claim("m", "d", [0, 1], "alice", "s1")
    theirs = led.claim("m", "d", [5, 6], "bob", "s2")
    done_mine = led.claim("m", "d2", [0, 1], "alice", "s1")
    led.complete(done_mine, "run", 2)

    keys = {(x.model, x.dataset, x.i0) for x in led.my_open("alice")}
    assert keys == {("m", "d", 0)}          # not bob's, not my already-done one
    _ = (mine, theirs)


def test_prune_done_only_removes_old_done_leases(tmp_path):
    led = Ledger(tmp_path / "claims")
    fresh_done = led.claim("m", "d", [0], "a", "s")
    led.complete(fresh_done, "r", 1)
    old_done = led.claim("m", "d", [1], "a", "s")
    led.complete(old_done, "r", 1)
    Lease  # noqa: B018 - import used for API surface

    # backdate the second one (indices [1] -> range 000001-000002)
    import json
    p = tmp_path / "claims" / _only_matching(tmp_path / "claims", "000001-000002")
    body = json.loads(p.read_text())
    body["updated_at"] = time.time() - 999999
    p.write_text(json.dumps(body))

    removed = led.prune_done(keep_sec=3600)
    assert removed == 1
    assert led.covered_indices("m", "d") == {0}   # the fresh done lease survives


def _only_matching(d, needle):
    return next(p.name for p in d.glob("*.json") if needle in p.name)
