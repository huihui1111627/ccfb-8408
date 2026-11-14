"""每日余额调节表。

调节逻辑（标准银行存款余额调节表，按账户出表）：
  银行对账单余额
    ＋ 企业已收、银行未收（账面应收未收齐部分，即未达账项中的应收侧）
    － 企业已付、银行未付（本 mock 场景无付款单，预留为 0）
  ＝ 调节后银行侧余额
  企业账面余额
    ＋ 银行已收、企业未收（未核销收入流水，含部分付款未配部分）
    － 银行已付、企业未付（未核销支出流水）
  ＝ 调节后企业侧余额
两边调节后相等即平衡。
"""
from __future__ import annotations

from .. import config, db


def build_report(conn, account_id: int, report_date: str) -> dict:
    acct = db.query_one(conn, "SELECT * FROM bank_account WHERE id=?", (account_id,))
    if acct is None:
        raise ValueError(f"账户不存在: {account_id}")

    snap = db.query_one(
        conn,
        "SELECT * FROM balance_snapshot WHERE account_id=? AND snapshot_date<=? "
        "ORDER BY snapshot_date DESC LIMIT 1",
        (account_id, report_date))
    if snap is None:
        # 报表日期早于任何一次下载：回退到最近可用快照（首次下载前无历史余额可取）
        snap = db.query_one(
            conn,
            "SELECT * FROM balance_snapshot WHERE account_id=? "
            "ORDER BY snapshot_date ASC LIMIT 1",
            (account_id,))
    bank_balance = snap["bank_balance"] if snap else 0.0

    # 银行已收企业未收：未配平的收入流水
    bank_add_rows = db.query_all(
        conn,
        "SELECT * FROM bank_tx WHERE account_id=? AND direction='in' AND txn_date<=? "
        "AND amount-matched_amount > ? ORDER BY txn_date, id",
        (account_id, report_date, config.AMOUNT_EPS))
    # 银行已付企业未付：未核销的支出流水（支出流水不参与应收核销，全额列示）
    bank_sub_rows = db.query_all(
        conn,
        "SELECT * FROM bank_tx WHERE account_id=? AND direction='out' AND txn_date<=? "
        "ORDER BY txn_date, id",
        (account_id, report_date))
    # 企业已收银行未收：该账户应收里未收齐的部分
    book_add_rows = db.query_all(
        conn,
        "SELECT * FROM receivable WHERE collect_account_id=? AND due_date<=? "
        "AND amount-received_amount > ? ORDER BY due_date, id",
        (account_id, report_date, config.AMOUNT_EPS))

    bank_add = round(sum(r["amount"] - r["matched_amount"] for r in bank_add_rows), 2)
    bank_sub = round(sum(r["amount"] for r in bank_sub_rows), 2)
    book_add = round(sum(r["amount"] - r["received_amount"] for r in book_add_rows), 2)
    book_sub = 0.0

    adjusted_bank = round(bank_balance + book_add - book_sub, 2)
    adjusted_book = round(acct["book_balance"] + bank_add - bank_sub, 2)
    balanced = abs(adjusted_bank - adjusted_book) <= config.AMOUNT_EPS

    return {
        "account": dict(acct),
        "report_date": report_date,
        "bank_balance": round(bank_balance, 2),
        "book_balance": round(acct["book_balance"], 2),
        "bank_add": bank_add,
        "bank_sub": bank_sub,
        "book_add": book_add,
        "book_sub": book_sub,
        "adjusted_bank": adjusted_bank,
        "adjusted_book": adjusted_book,
        "balanced": balanced,
        "bank_add_rows": [dict(r) for r in bank_add_rows],
        "bank_sub_rows": [dict(r) for r in bank_sub_rows],
        "book_add_rows": [dict(r) for r in book_add_rows],
    }
