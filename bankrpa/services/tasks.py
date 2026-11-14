"""下载任务服务：调度 RPA 引擎、流水幂等入库、增量下载与跨天补漏。

红线1 在本模块落地：bank_tx UNIQUE(bank_code, txn_ref) + INSERT OR IGNORE，
同一笔流水无论任务跑多少次、并发跑多少个，都只会入库一次。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from .. import config, db
from ..rpa.config_loader import get_bank_config
from ..rpa.engine import run_download


def create_task(conn, account_id: int, date_from: str, date_to: str) -> int:
    acct = db.query_one(conn, "SELECT * FROM bank_account WHERE id=?", (account_id,))
    if acct is None:
        raise ValueError(f"账户不存在: {account_id}")
    cur = conn.execute(
        "INSERT INTO download_task(account_id,date_from,date_to,status,created_at) "
        "VALUES(?,?,?,'pending',?)",
        (account_id, date_from, date_to, db.now_iso()),
    )
    conn.commit()
    db.insert_audit(conn, "create_task", "download_task", cur.lastrowid,
                    f"账户 {acct['account_no']} {date_from}~{date_to}")
    conn.commit()
    return cur.lastrowid


def latest_success_range(conn, account_id: int) -> tuple[str, str] | None:
    row = db.query_one(
        conn,
        "SELECT date_from, date_to FROM download_task "
        "WHERE account_id=? AND status='success' ORDER BY finished_at DESC LIMIT 1",
        (account_id,))
    return (row["date_from"], row["date_to"]) if row else None


def create_incremental_task(conn, account_id: int, today: str | None = None) -> int:
    """增量下载：取上次成功任务的结束日次日到今天；没有历史则从 7 天前开始。

    任务区间与历史区间重叠也没关系——INSERT OR IGNORE 保证重复流水不重复入库，
    这正是跨天补漏可以安全重跑的基础。
    """
    today = today or date.today().isoformat()
    last = latest_success_range(conn, account_id)
    if last:
        start = (datetime.fromisoformat(last[1]) + timedelta(days=1)).date().isoformat()
        if start > today:
            start = today
    else:
        start = (datetime.fromisoformat(today) - timedelta(days=7)).date().isoformat()
    return create_task(conn, account_id, start, today)


def create_catchup_task(conn, account_id: int, gap_from: str, gap_to: str) -> int:
    """跨天补漏：对指定历史区间显式重跑，重复流水被唯一约束挡掉。"""
    return create_task(conn, account_id, gap_from, gap_to)


def _upsert_balance_snapshot(conn, account_id: int, snapshot_date: str, balance: float) -> None:
    conn.execute(
        "INSERT INTO balance_snapshot(account_id,snapshot_date,bank_balance) VALUES(?,?,?) "
        "ON CONFLICT(account_id,snapshot_date) DO UPDATE SET bank_balance=excluded.bank_balance",
        (account_id, snapshot_date, balance),
    )


def run_task(conn, task_id: int, headless: bool = True) -> dict:
    task = db.query_one(conn, "SELECT * FROM download_task WHERE id=?", (task_id,))
    if task is None:
        raise ValueError(f"任务不存在: {task_id}")
    if task["status"] == "running":
        raise RuntimeError(f"任务 {task_id} 正在运行，禁止并发执行同一任务")

    acct = db.query_one(conn, "SELECT * FROM bank_account WHERE id=?", (task["account_id"],))
    conn.execute("UPDATE download_task SET status='running',started_at=? WHERE id=?",
                 (db.now_iso(), task_id))
    conn.commit()
    try:
        cfg = get_bank_config(acct["bank_code"])
        result = run_download(
            base_url=acct["base_url"], bank_code=acct["bank_code"], cfg=cfg,
            username=acct["login_user"], password=acct["login_pwd"],
            date_from=task["date_from"], date_to=task["date_to"],
            token_url=config.TOKEN_URL, headless=headless,
        )
        inserted = 0
        for txn in result["txns"]:
            cur = conn.execute(
                "INSERT OR IGNORE INTO bank_tx("
                "account_id,bank_code,txn_ref,txn_time,txn_date,direction,amount,"
                "counterparty,summary,receipt_path,receipt_payer,receipt_purpose,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (acct["id"], acct["bank_code"], txn["txn_ref"], txn["txn_time"],
                 txn["txn_date"], txn["direction"], txn["amount"], txn.get("counterparty"),
                 txn.get("summary"), txn.get("receipt_path"), txn.get("receipt_payer"),
                 txn.get("receipt_purpose"), db.now_iso()),
            )
            inserted += cur.rowcount
        if result.get("balance") is not None:
            _upsert_balance_snapshot(conn, acct["id"], task["date_to"], result["balance"])
        conn.execute(
            "UPDATE download_task SET status='success',rows_fetched=?,rows_inserted=?,"
            "receipts_saved=?,message=?,finished_at=? WHERE id=?",
            (result["rows_fetched"], inserted, result["receipts_saved"],
             f"抓取 {result['rows_fetched']} 笔，新增 {inserted} 笔（重复已跳过）",
             db.now_iso(), task_id),
        )
        db.insert_audit(conn, "run_task_success", "download_task", task_id,
                        f"抓取 {result['rows_fetched']} 新增 {inserted} 回单 {result['receipts_saved']}")
        conn.commit()
        return {"rows_fetched": result["rows_fetched"], "rows_inserted": inserted,
                "receipts_saved": result["receipts_saved"]}
    except Exception as exc:
        conn.rollback()
        conn.execute(
            "UPDATE download_task SET status='failed',message=?,finished_at=? WHERE id=?",
            (f"{type(exc).__name__}: {exc}", db.now_iso(), task_id),
        )
        db.insert_audit(conn, "run_task_failed", "download_task", task_id, str(exc))
        conn.commit()
        raise
