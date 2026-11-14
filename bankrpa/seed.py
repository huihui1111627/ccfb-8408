"""初始化数据库并灌入 mock 种子数据（账户 + 应收台账）。

流水由 RPA 任务真实跑浏览器下载入库，本脚本不直接插流水，
保证演示路径与生产路径完全一致。可重复执行（UPSERT 幂等）。
"""
from __future__ import annotations

from . import config, db
from .mock_bank.data import BANKS, BOOK_BALANCES, RECEIVABLES


def seed(db_path=None) -> None:
    config.ensure_dirs()
    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        account_ids: dict[str, int] = {}
        for idx, (code, bank) in enumerate(BANKS.items()):
            port = config.BANK_PORTS[idx]
            existing = db.query_one(conn, "SELECT id FROM bank_account WHERE bank_code=?", (code,))
            if existing:
                account_ids[code] = existing["id"]
                conn.execute(
                    "UPDATE bank_account SET base_url=?, book_balance=? WHERE id=?",
                    (f"http://127.0.0.1:{port}", BOOK_BALANCES[code], existing["id"]))
                continue
            cur = conn.execute(
                "INSERT INTO bank_account(bank_code,bank_name,account_no,account_name,base_url,"
                "login_user,login_pwd,book_balance,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (code, bank["bank_name"], bank["account_no"], bank["account_name"],
                 f"http://127.0.0.1:{port}", bank["login_user"], bank["login_pwd"],
                 BOOK_BALANCES[code], db.now_iso()))
            account_ids[code] = cur.lastrowid

        for ar in RECEIVABLES:
            acct_id = account_ids.get(ar["bank"])
            existing = db.query_one(conn, "SELECT id FROM receivable WHERE ar_no=?", (ar["ar_no"],))
            if existing:
                continue
            conn.execute(
                "INSERT INTO receivable(ar_no,customer,amount,received_amount,"
                "collect_account_id,due_date,created_at) VALUES(?,?,?,0,?,?,?)",
                (ar["ar_no"], ar["customer"], ar["amount"], acct_id,
                 ar["due_date"], db.now_iso()))

        db.insert_audit(conn, "seed", "bank_account", None, "初始化 mock 账户与应收台账")
        conn.commit()
        print(f"种子数据完成：{len(account_ids)} 个账户，{len(RECEIVABLES)} 条应收，库={config.DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
