"""自动对账：1:1、1:N、N:1、部分付款/配不上落差异，以及余额调节表。"""
from __future__ import annotations

import pytest

from bankrpa import db
from bankrpa.services import reconcile, report as report_svc, tasks
from bankrpa.services.reconcile import MatchError


@pytest.fixture()
def downloaded(conn, mock_servers, account_id):
    for code in ("ecd", "scb", "wcb"):
        tid = tasks.create_task(conn, account_id(code), "2026-11-12", "2026-11-13")
        tasks.run_task(conn, tid, headless=True)
    return account_id


def test_auto_match_covers_all_patterns(conn, downloaded):
    stats = reconcile.auto_match(conn)
    # 1:1：R001(10000)、R002(3500)、R006(6000)
    # 1:N：7000 合并付款拆 R008(3000)+R009(4000)，共 2 条
    # N:1：R005 两笔(2000+3000)、R007 两笔(2500+1500)，共 4 条
    # 部分付款：R004 来 5500、R010 来 2000，共 2 条
    assert stats["one_to_one"] == 3
    assert stats["one_to_many"] == 2
    assert stats["many_to_one"] == 4
    assert stats["partial"] == 2

    # 已收齐的应收
    for no in ("R001", "R002", "R005", "R006", "R007", "R008", "R009"):
        ar = db.query_one(conn, "SELECT * FROM receivable WHERE ar_no=?", (no,))
        assert abs(ar["received_amount"] - ar["amount"]) < 0.01, no

    # 合并付款流水 7000 应被两张应收完全吃掉
    t = db.query_one(conn, "SELECT * FROM bank_tx WHERE txn_ref='WCB-20261112-003'")
    assert abs(t["matched_amount"] - 7000.0) < 0.01

    # N:1 两张流水各自的已配金额
    t1 = db.query_one(conn, "SELECT * FROM bank_tx WHERE txn_ref='WCB-20261113-001'")
    t2 = db.query_one(conn, "SELECT * FROM bank_tx WHERE txn_ref='WCB-20261113-002'")
    assert t1["matched_amount"] == 2500.0 and t2["matched_amount"] == 1500.0


def test_partial_and_unmatched_go_to_diff(conn, downloaded):
    reconcile.auto_match(conn)
    open_diffs = db.query_all(
        conn, "SELECT * FROM diff_item WHERE status='open' ORDER BY id")
    kinds = {(d["kind"], d["ref_id"]) for d in open_diffs}

    # R004 部分付款：来 5500 欠 6500 -> 应收差异
    r004 = db.query_one(conn, "SELECT id FROM receivable WHERE ar_no='R004'")
    assert ("unmatched_ar", r004["id"]) in kinds
    d = next(d for d in open_diffs if d["ref_id"] == r004["id"])
    assert d["amount"] == pytest.approx(6500.0)

    # R003(8200)、R010(3800 只来 2000) 未收齐
    for no in ("R003", "R010"):
        rid = db.query_one(conn, "SELECT id FROM receivable WHERE ar_no=?", (no,))["id"]
        assert ("unmatched_ar", rid) in kinds

    # 完全配不上的收入流水：苏州 4800、无锡 900
    for ref in ("ECD-20261113-002", "SCB-20261112-005"):
        tid = db.query_one(conn, "SELECT id FROM bank_tx WHERE txn_ref=?", (ref,))["id"]
        assert ("unmatched_txn", tid) in kinds


def test_manual_resolve_diff_with_trail(conn, downloaded):
    reconcile.auto_match(conn)
    d = db.query_one(conn, "SELECT * FROM diff_item WHERE status='open' LIMIT 1")
    reconcile.resolve_diff_manually(conn, d["id"], "客户已承诺下周补款，挂账观察", "cfo01")
    again = db.query_one(conn, "SELECT * FROM diff_item WHERE id=?", (d["id"],))
    assert again["status"] == "resolved" and again["operator"] == "cfo01"
    with pytest.raises(MatchError):
        reconcile.resolve_diff_manually(conn, d["id"], "重复处理", "x")
    log = db.query_one(conn, "SELECT * FROM audit_log WHERE action='resolve_diff'")
    assert log is not None


def test_resolve_requires_reason(conn, downloaded):
    reconcile.auto_match(conn)
    d = db.query_one(conn, "SELECT * FROM diff_item WHERE status='open' LIMIT 1")
    with pytest.raises(MatchError):
        reconcile.resolve_diff_manually(conn, d["id"], "  ")


def test_balance_report_balanced_after_full_manual(conn, downloaded, account_id):
    reconcile.auto_match(conn)
    # 把剩余差异全部人工核销掉（业务上确认无需再追）
    for d in db.query_all(conn, "SELECT * FROM diff_item WHERE status='open'"):
        reconcile.resolve_diff_manually(conn, d["id"], "人工确认处理", "cfo01")

    rep = report_svc.build_report(conn, account_id("ecd"), "2026-11-13")
    # 调节表结构完整
    assert {"bank_balance", "book_balance", "adjusted_bank",
            "adjusted_book", "balanced"} <= set(rep)
    # 银行已付企业未付 = ecd 那笔 3000 广告费
    assert rep["bank_sub"] == pytest.approx(3000.0)
    assert rep["bank_balance"] == pytest.approx(158600.0)


def test_report_reflects_unmatched_amounts(conn, downloaded, account_id):
    reconcile.auto_match(conn)
    rep_scb = report_svc.build_report(conn, account_id("scb"), "2026-11-13")
    # 900 保证金配不上，属于银行已收企业未收
    assert rep_scb["bank_add"] == pytest.approx(900.0)
    # 调节后两侧金额自洽（种子账面余额按调节公式反推，自动对账后必然平衡）
    assert rep_scb["adjusted_bank"] == pytest.approx(rep_scb["adjusted_book"])
    assert rep_scb["balanced"] is True


def test_auto_match_idempotent_rerun(conn, downloaded):
    s1 = reconcile.auto_match(conn)
    matched_after_first = db.query_one(
        conn, "SELECT COUNT(*) c FROM match_entry WHERE status='active'")["c"]
    s2 = reconcile.auto_match(conn)  # 再跑一次不应新增核销
    matched_after_second = db.query_one(
        conn, "SELECT COUNT(*) c FROM match_entry WHERE status='active'")["c"]
    assert matched_after_first == matched_after_second
    assert s2["one_to_one"] == 0 and s2["one_to_many"] == 0 \
        and s2["many_to_one"] == 0 and s2["partial"] == 0
