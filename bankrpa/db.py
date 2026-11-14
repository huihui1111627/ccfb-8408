"""SQLite 数据访问层。

两条硬性幂等红线在这里由数据库约束兜底：
1. 同一笔流水不重复入库：bank_tx UNIQUE(bank_code, txn_ref) + INSERT OR IGNORE。
2. 同一笔流水/应收不重复核销：match_entry 的条件唯一索引
   (txn_id, receivable_id) WHERE status='active'，配合核销事务里的
   BEGIN IMMEDIATE 与剩余余额校验，并发下只允许一笔成功。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from . import config

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS bank_account(
    id INTEGER PRIMARY KEY,
    bank_code TEXT NOT NULL UNIQUE,
    bank_name TEXT NOT NULL,
    account_no TEXT NOT NULL,
    account_name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    login_user TEXT NOT NULL,
    login_pwd TEXT NOT NULL,
    book_balance REAL NOT NULL DEFAULT 0,   -- 企业账面余额（期初）
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS download_task(
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES bank_account(id),
    date_from TEXT NOT NULL,
    date_to TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending/running/success/failed
    rows_fetched INTEGER NOT NULL DEFAULT 0,
    rows_inserted INTEGER NOT NULL DEFAULT 0,
    receipts_saved INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_account ON download_task(account_id);

CREATE TABLE IF NOT EXISTS bank_tx(
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES bank_account(id),
    bank_code TEXT NOT NULL,
    txn_ref TEXT NOT NULL,                 -- 银行侧唯一流水号
    txn_time TEXT NOT NULL,
    txn_date TEXT NOT NULL,
    direction TEXT NOT NULL,               -- in / out
    amount REAL NOT NULL,
    counterparty TEXT,
    summary TEXT,
    receipt_path TEXT,
    receipt_payer TEXT,
    receipt_purpose TEXT,
    matched_amount REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(bank_code, txn_ref)             -- 红线1：同一笔流水不能入库两次
);
CREATE INDEX IF NOT EXISTS idx_tx_account_date ON bank_tx(account_id, txn_date);

CREATE TABLE IF NOT EXISTS receivable(
    id INTEGER PRIMARY KEY,
    ar_no TEXT NOT NULL UNIQUE,
    customer TEXT NOT NULL,
    amount REAL NOT NULL,
    received_amount REAL NOT NULL DEFAULT 0,
    collect_account_id INTEGER REFERENCES bank_account(id),  -- 预期收款账户，可空
    due_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS match_entry(
    id INTEGER PRIMARY KEY,
    txn_id INTEGER NOT NULL REFERENCES bank_tx(id),
    receivable_id INTEGER NOT NULL REFERENCES receivable(id),
    amount REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', -- active / reversed
    source TEXT NOT NULL,                  -- auto / manual
    operator TEXT,
    created_at TEXT NOT NULL,
    reversed_at TEXT
);
-- 红线2：同一对(流水,应收)同时只能有一条有效核销
CREATE UNIQUE INDEX IF NOT EXISTS uq_match_active
    ON match_entry(txn_id, receivable_id) WHERE status='active';

CREATE TABLE IF NOT EXISTS diff_item(
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                    -- unmatched_txn / unmatched_ar
    ref_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'open',   -- open / resolved
    resolution TEXT,
    operator TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_diff_open
    ON diff_item(kind, ref_id) WHERE status='open';

CREATE TABLE IF NOT EXISTS audit_log(
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_id INTEGER,
    detail TEXT,
    operator TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS balance_snapshot(
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES bank_account(id),
    snapshot_date TEXT NOT NULL,
    bank_balance REAL NOT NULL,
    UNIQUE(account_id, snapshot_date)
);
"""


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def immediate_tx(conn: sqlite3.Connection):
    """立即拿写锁的事务，核销/反转等关键写操作统一走这里，杜绝并发双花。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def insert_audit(conn: sqlite3.Connection, action: str, entity: str,
                 entity_id: int | None, detail: str, operator: str = "system") -> None:
    conn.execute(
        "INSERT INTO audit_log(action,entity,entity_id,detail,operator,created_at) "
        "VALUES(?,?,?,?,?,?)",
        (action, entity, entity_id, detail, operator, now_iso()),
    )


def query_all(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchall()


def query_one(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, tuple(params)).fetchone()
