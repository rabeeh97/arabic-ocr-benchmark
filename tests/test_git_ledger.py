"""GitLedger: leases coordinated over a git branch instead of a shared dir.

A bare repo stands in for GitHub; two clones stand in for two Colab runners on
different Google accounts (which do NOT share a Drive tree).
"""

from __future__ import annotations

import subprocess

import pytest

from arabic_ocr_bench.coordination import GitLedger, runner_identity


def _run(*args: str, cwd=None) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def two_runners(tmp_path):
    origin = tmp_path / "origin.git"
    _run("git", "init", "--bare", "-b", "main", str(origin))
    clone_a = tmp_path / "runner_a"
    clone_b = tmp_path / "runner_b"
    _run("git", "clone", str(origin), str(clone_a))
    _run("git", "clone", str(origin), str(clone_b))
    a = GitLedger(clone_a, branch="sweep-claims", ttl_sec=2, sync_every=0.0)
    b = GitLedger(clone_b, branch="sweep-claims", ttl_sec=2, sync_every=0.0)
    return a, b


def test_a_claim_is_visible_to_b_and_blocks_an_overlap(two_runners):
    a, b = two_runners
    rid_a, sid_a = runner_identity("runner-a")
    rid_b, sid_b = runner_identity("runner-b")

    lease_a = a.claim("qwen2.5-vl-3b", "muharaf_lines", list(range(10)), rid_a, sid_a)
    assert lease_a is not None

    # b sees a's claim through the branch, without any shared filesystem
    assert b.covered_indices("qwen2.5-vl-3b", "muharaf_lines") == set(range(10))

    # b tries the same rows — a claimed first, so a wins the tiebreak and b backs off
    assert b.claim("qwen2.5-vl-3b", "muharaf_lines", list(range(5, 15)), rid_b, sid_b) is None


def test_disjoint_ranges_both_succeed(two_runners):
    a, b = two_runners
    rid_a, sid_a = runner_identity("runner-a")
    rid_b, sid_b = runner_identity("runner-b")

    assert a.claim("m", "d", list(range(10)), rid_a, sid_a) is not None
    lease_b = b.claim("m", "d", list(range(10, 20)), rid_b, sid_b)
    assert lease_b is not None

    cov = a.covered_indices("m", "d")
    assert cov == set(range(20))


def test_completed_lease_still_covers_and_is_seen_by_the_other_runner(two_runners):
    a, b = two_runners
    rid_a, sid_a = runner_identity("runner-a")
    lease_a = a.claim("m", "d", list(range(10)), rid_a, sid_a)
    a.complete(lease_a, "20260905T120000Z", 10)

    b_leases = b._load_all()
    assert [x.status for x in b_leases] == ["done"]
    assert b.covered_indices("m", "d") == set(range(10))


def test_expired_lease_is_reclaimable_by_the_other_runner(two_runners):
    import time

    a, b = two_runners
    rid_a, sid_a = runner_identity("runner-a")
    rid_b, sid_b = runner_identity("runner-b")
    a.claim("m", "d", list(range(10)), rid_a, sid_a)

    time.sleep(2.2)  # ttl_sec=2
    stale = b.reclaimable()
    assert len(stale) == 1
    taken = b.adopt(stale[0], rid_b, sid_b)
    assert taken is not None and b.confirm(taken)
    assert taken.runner_id == rid_b


def test_two_runners_racing_the_same_batch_only_one_proceeds(two_runners):
    """The exact collision seen in the wild: both runners pick
    qwen2.5-vl-3b / muharaf_lines rows 0..9 at once."""
    a, b = two_runners
    rid_a, sid_a = runner_identity("runner-a")
    rid_b, sid_b = runner_identity("runner-b")
    rows = list(range(10))

    lease_a = a.claim("qwen2.5-vl-3b", "muharaf_lines", rows, rid_a, sid_a)
    lease_b = b.claim("qwen2.5-vl-3b", "muharaf_lines", rows, rid_b, sid_b)

    proceed = [(lease is not None) and led.confirm(lease)
               for led, lease in ((a, lease_a), (b, lease_b))]
    assert proceed.count(True) == 1  # exactly one runner does the work


def test_release_removes_the_claim_for_everyone(two_runners):
    a, b = two_runners
    rid_a, sid_a = runner_identity("runner-a")
    rid_b, sid_b = runner_identity("runner-b")

    lease_a = a.claim("m", "d", list(range(10)), rid_a, sid_a)
    a.release(lease_a)

    assert b.covered_indices("m", "d") == set()
    # the rows are free again — b can take them
    assert b.claim("m", "d", list(range(10)), rid_b, sid_b) is not None
