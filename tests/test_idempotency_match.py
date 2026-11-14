"""红线2：同一笔流水/应收不能重复核销，含并发双花测试与全部配对形态。"""
from __future__ import annotations

import threading

import pytest

from bankrpa import config, db
from bankrpa.services import reconcile, tasks


@pytest.fixture()
def downloaded(conn, mock_servers, account_id):
    """跑完全部三家下载，返回 (txn 映射, 应收 映射)。"""
    for code in ("ecd", "scb", "wcb"):
        tid = tasks.create_task(conn, account_id(code), "2026-11-12", "2026-11-13")
        tasks.run_task(conn, tid, headless=True)

    def txn(ref: str):
        return db.query_one(conn, "SELECT * FROM bank_tx WHERE txn_ref=?", (ref,))

    def ar(no: str):
        return db.query_one(conn, "SELECT * FROM receivable WHERE ar_no=?", (no,))

    return txn, ar


def test_simple_writeoff_and_balances(conn, downloaded):
    txn, ar = downloaded
    mid = reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10000.0)
    assert mid > 0
    t = txn("ECD-20261113-001")
    a = ar("R001")
    assert t["matched_amount"] == 10000.0
    assert a["received_amount"] == 10000.0


def test_duplicate_pair_rejected(conn, downloaded):
    txn, ar = downloaded
    reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10000.0)
    with pytest.raises(reconcile.MatchError):
        reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10000.0)


def test_over_amount_rejected(conn, downloaded):
    txn, ar = downloaded
    # 流水只有 10000，核销 10001 必须被余额校验挡住
    with pytest.raises(reconcile.MatchError):
        reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10001.0)
    # 应收只剩 10000，对同一张应收二次超额也挡住
    reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10000.0)
    other_txn = txn("SCB-20261112-004")
    with pytest.raises(reconcile.MatchError):
        reconcile.writeoff(conn, other_txn["id"], ar("R001")["id"], 1.0)


def test_partial_payment_creates_diff(conn, downloaded):
    txn, ar = downloaded
    # R004 应收 12000，流水只来 5500（部分付款）
    reconcile.writeoff(conn, txn("ECD-20261113-004")["id"], ar("R004")["id"], 5500.0)
    d = db.query_one(
        conn, "SELECT * FROM diff_item WHERE kind='unmatched_ar' AND ref_id=? AND status='open'",
        (ar("R004")["id"],))
    assert d is not None and d["amount"] == pytest.approx(6500.0)
    # 流水已配平，不应再有流水侧差异
    dt = db.query_one(
        conn, "SELECT * FROM diff_item WHERE kind='unmatched_txn' AND ref_id=? AND status='open'",
        (txn("ECD-20261113-004")["id"],))
    assert dt is None


def test_concurrent_writeoff_only_one_wins(conn, db_path, downloaded):
    """并发对同一对流水/应收核销，只有一笔成功，其余拿锁后余额不足失败。"""
    txn, ar = downloaded
    txn_id, ar_id = txn("ECD-20261113-001")["id"], ar("R001")["id"]
    results: list[str] = []
    barrier = threading.Barrier(8)

    def worker():
        c = db.connect(db_path)
        barrier.wait()  # 尽量同时冲锁
        try:
            reconcile.writeoff(c, txn_id, ar_id, 10000.0)
            results.append("ok")
        except reconcile.MatchError:
            results.append("fail")
        finally:
            c.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") == 1
    assert results.count("fail") == 7
    # 库里也只能有一条有效核销，已配金额恰好 10000
    assert db.query_one(
        conn,
        "SELECT COUNT(*) c FROM match_entry WHERE txn_id=? AND receivable_id=? AND status='active'",
        (txn_id, ar_id))["c"] == 1
    assert db.query_one(conn, "SELECT matched_amount FROM bank_tx WHERE id=?",
                        (txn_id,))["matched_amount"] == 10000.0


def test_reverse_writeoff_releases_balances(conn, downloaded):
    txn, ar = downloaded
    mid = reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10000.0)
    reconcile.reverse_writeoff(conn, mid)
    assert txn("ECD-20261113-001")["matched_amount"] == 0.0
    assert ar("R001")["received_amount"] == 0.0
    # 冲销后可以重新核销
    reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"], 10000.0)
    # 重复冲销被拒绝
    with pytest.raises(reconcile.MatchError):
        reconcile.reverse_writeoff(conn, mid)


def test_outgoing_txn_cannot_writeoff(conn, downloaded):
    txn, ar = downloaded
    with pytest.raises(reconcile.MatchError):
        reconcile.writeoff(conn, txn("ECD-20261112-003")["id"], ar("R001")["id"], 1.0)


def test_audit_trail_present(conn, downloaded):
    txn, ar = downloaded
    reconcile.writeoff(conn, txn("ECD-20261113-001")["id"], ar("R001")["id"],
                       10000.0, operator="tester")
    log = db.query_one(
        conn, "SELECT * FROM audit_log WHERE action='writeoff' AND operator='tester'")
    assert log is not None and "R001" in log["detail"]
