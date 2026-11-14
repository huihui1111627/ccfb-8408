"""红线1：同一笔流水不能重复入库（含重复任务、跨天补漏、并发场景）。"""
from __future__ import annotations

import threading

import pytest

from bankrpa import db
from bankrpa.services import tasks


def _run_all_banks(conn, mock_servers, account_id, date_from="2026-11-12",
                   date_to="2026-11-13"):
    ids = []
    for code in ("ecd", "scb", "wcb"):
        tid = tasks.create_task(conn, account_id(code), date_from, date_to)
        ids.append(tid)
    for tid in ids:
        tasks.run_task(conn, tid, headless=True)


def test_full_download_idempotent(conn, mock_servers, account_id):
    _run_all_banks(conn, mock_servers, account_id)
    total_first = db.query_one(conn, "SELECT COUNT(*) c FROM bank_tx")["c"]
    assert total_first == 13  # 4 + 5 + 4

    # 同样的任务再跑一遍：抓取数不变，新增数必须为 0
    for code in ("ecd", "scb", "wcb"):
        tid = tasks.create_task(conn, account_id(code), "2026-11-12", "2026-11-13")
        stats = tasks.run_task(conn, tid, headless=True)
        assert stats["rows_fetched"] > 0
        assert stats["rows_inserted"] == 0
    assert db.query_one(conn, "SELECT COUNT(*) c FROM bank_tx")["c"] == total_first


def test_catchup_rerun_safe(conn, mock_servers, account_id):
    # 先只下 13 日，再补 12 日（区间重叠/补漏），最后全量重跑
    tid = tasks.create_task(conn, account_id("ecd"), "2026-11-13", "2026-11-13")
    tasks.run_task(conn, tid, headless=True)
    n1 = db.query_one(conn, "SELECT COUNT(*) c FROM bank_tx")["c"]

    tid = tasks.create_catchup_task(conn, account_id("ecd"), "2026-11-12", "2026-11-13")
    tasks.run_task(conn, tid, headless=True)
    n2 = db.query_one(conn, "SELECT COUNT(*) c FROM bank_tx")["c"]
    assert n2 > n1  # 补到了 12 日的流水

    tid = tasks.create_task(conn, account_id("ecd"), "2026-11-12", "2026-11-13")
    stats = tasks.run_task(conn, tid, headless=True)
    assert stats["rows_inserted"] == 0  # 补漏重跑不产生重复


def test_incremental_task_window(conn, mock_servers, account_id):
    # 无历史时回溯 7 天
    tid = tasks.create_incremental_task(conn, account_id("wcb"), today="2026-11-13")
    task = db.query_one(conn, "SELECT * FROM download_task WHERE id=?", (tid,))
    assert task["date_from"] == "2026-11-06" and task["date_to"] == "2026-11-13"
    tasks.run_task(conn, tid, headless=True)

    # 有历史后从结束日次日开始
    tid2 = tasks.create_incremental_task(conn, account_id("wcb"), today="2026-11-20")
    task2 = db.query_one(conn, "SELECT * FROM download_task WHERE id=?", (tid2,))
    assert task2["date_from"] == "2026-11-14" and task2["date_to"] == "2026-11-20"


def test_concurrent_same_ref_insert_only_one(conn, db_path):
    """直接对同一 (bank_code, txn_ref) 并发 INSERT，只有一条能成功。"""
    acct = db.query_one(conn, "SELECT id FROM bank_account LIMIT 1")["id"]
    errors: list[Exception] = []

    def insert():
        c = db.connect(db_path)
        try:
            c.execute(
                "INSERT OR IGNORE INTO bank_tx(account_id,bank_code,txn_ref,txn_time,"
                "txn_date,direction,amount,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (acct, "ecd", "CONC-1", "2026-11-13 10:00:00", "2026-11-13",
                 "in", 1.0, db.now_iso()))
            c.commit()
        except Exception as exc:  # pragma: no cover - 防御性
            errors.append(exc)
        finally:
            c.close()

    threads = [threading.Thread(target=insert) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert db.query_one(
        conn, "SELECT COUNT(*) c FROM bank_tx WHERE txn_ref='CONC-1'")["c"] == 1


def test_running_task_cannot_run_twice(conn, mock_servers, account_id):
    tid = tasks.create_task(conn, account_id("ecd"), "2026-11-13", "2026-11-13")
    conn.execute("UPDATE download_task SET status='running' WHERE id=?", (tid,))
    conn.commit()
    with pytest.raises(RuntimeError):
        tasks.run_task(conn, tid, headless=True)


def test_empty_window_is_success_not_failure(conn, mock_servers, account_id):
    """增量窗口落在没有流水的日期：任务成功、抓取 0 笔，而不是超时失败。"""
    tid = tasks.create_task(conn, account_id("scb"), "2026-11-14", "2026-11-14")
    stats = tasks.run_task(conn, tid, headless=True)
    assert stats["rows_fetched"] == 0 and stats["rows_inserted"] == 0
    task = db.query_one(conn, "SELECT * FROM download_task WHERE id=?", (tid,))
    assert task["status"] == "success"


def test_balance_snapshot_upsert(conn, mock_servers, account_id):
    tid = tasks.create_task(conn, account_id("ecd"), "2026-11-13", "2026-11-13")
    tasks.run_task(conn, tid, headless=True)
    snap = db.query_one(
        conn, "SELECT * FROM balance_snapshot WHERE account_id=? AND snapshot_date='2026-11-13'",
        (account_id("ecd"),))
    assert snap is not None and snap["bank_balance"] == 158600.0
