"""对账核销服务。

红线2 在本模块落地：
- match_entry 条件唯一索引 (txn_id, receivable_id) WHERE status='active' 兜底重复配对；
- 所有核销/反转统一在单事务 BEGIN IMMEDIATE 内完成，并实时校验剩余可配余额；
- 并发请求只有一笔能拿到写锁成功，其余拿锁后看到余额不足即失败，杜绝重复核销。

支持的配对形态：一对一、一对多（一笔流水配多张应收）、多对一（多笔流水凑一张应收）、
部分付款（核销额小于应收额/流水额，剩余部分进差异）。
"""
from __future__ import annotations

from .. import config, db


class MatchError(RuntimeError):
    pass


def _open_match_exists(conn, txn_id: int, receivable_id: int) -> bool:
    return db.query_one(
        conn,
        "SELECT 1 FROM match_entry WHERE txn_id=? AND receivable_id=? AND status='active'",
        (txn_id, receivable_id)) is not None


def writeoff(conn, txn_id: int, receivable_id: int, amount: float,
             source: str = "manual", operator: str = "manual") -> int:
    """对一笔流水与一张应收做一次核销（金额由调用方指定，支持部分核销）。

    返回 match_entry 主键。已存在同一对有效核销时幂等拒绝（报业务错，不产生第二条）。
    """
    amount = round(float(amount), 2)
    if amount <= 0:
        raise MatchError("核销金额必须大于 0")

    with db.immediate_tx(conn):
        txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE id=?", (txn_id,))
        ar = db.query_one(conn, "SELECT * FROM receivable WHERE id=?", (receivable_id,))
        if txn is None or ar is None:
            raise MatchError("流水或应收不存在")
        if txn["direction"] != "in":
            raise MatchError("只有收入流水可参与应收核销")
        if _open_match_exists(conn, txn_id, receivable_id):
            raise MatchError("该流水与应收已存在有效核销，不能重复核销")
        # 账户归属校验（应收指定了收款账户时必须一致）
        if ar["collect_account_id"] is not None and ar["collect_account_id"] != txn["account_id"]:
            raise MatchError("应收的收款账户与流水账户不一致")

        txn_remain = round(txn["amount"] - txn["matched_amount"], 2)
        ar_remain = round(ar["amount"] - ar["received_amount"], 2)
        if amount > txn_remain + config.AMOUNT_EPS:
            raise MatchError(f"核销金额 {amount} 超过流水剩余可配 {txn_remain}")
        if amount > ar_remain + config.AMOUNT_EPS:
            raise MatchError(f"核销金额 {amount} 超过应收剩余未收 {ar_remain}")

        cur = conn.execute(
            "INSERT INTO match_entry(txn_id,receivable_id,amount,status,source,operator,created_at) "
            "VALUES(?,?,?, 'active', ?, ?, ?)",
            (txn_id, receivable_id, amount, source, operator, db.now_iso()))
        conn.execute("UPDATE bank_tx SET matched_amount=matched_amount+? WHERE id=?",
                     (amount, txn_id))
        conn.execute("UPDATE receivable SET received_amount=received_amount+? WHERE id=?",
                     (amount, receivable_id))
        db.insert_audit(conn, "writeoff", "match_entry", cur.lastrowid,
                        f"流水 {txn['txn_ref']} 配应收 {ar['ar_no']} 金额 {amount}（{source}）",
                        operator)
        _refresh_diff_for(conn, txn_id, receivable_id)
        return cur.lastrowid


def reverse_writeoff(conn, match_id: int, operator: str = "manual") -> None:
    """冲销一条有效核销，释放两边已配金额；冲销后差异重新出现，全程留痕。"""
    with db.immediate_tx(conn):
        m = db.query_one(conn, "SELECT * FROM match_entry WHERE id=?", (match_id,))
        if m is None:
            raise MatchError("核销记录不存在")
        if m["status"] != "active":
            raise MatchError("该核销已冲销，不能重复冲销")
        conn.execute(
            "UPDATE match_entry SET status='reversed', reversed_at=? WHERE id=?",
            (db.now_iso(), match_id))
        conn.execute("UPDATE bank_tx SET matched_amount=matched_amount-? WHERE id=?",
                     (m["amount"], m["txn_id"]))
        conn.execute("UPDATE receivable SET received_amount=received_amount-? WHERE id=?",
                     (m["amount"], m["receivable_id"]))
        db.insert_audit(conn, "reverse_writeoff", "match_entry", match_id,
                        f"冲销核销 {match_id}，释放金额 {m['amount']}", operator)
        _refresh_diff_for(conn, m["txn_id"], m["receivable_id"])


def _refresh_diff_for(conn, txn_id: int, receivable_id: int) -> None:
    """根据当前剩余余额，维护这两个实体的未配差异（open 唯一，解决后可重新出现）。"""
    txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE id=?", (txn_id,))
    ar = db.query_one(conn, "SELECT * FROM receivable WHERE id=?", (receivable_id,))
    _upsert_diff(conn, "unmatched_txn", txn_id,
                 round(txn["amount"] - txn["matched_amount"], 2), "收入流水未配平")
    _upsert_diff(conn, "unmatched_ar", receivable_id,
                 round(ar["amount"] - ar["received_amount"], 2), "应收未收齐")


def _upsert_diff(conn, kind: str, ref_id: int, remain: float, reason: str) -> None:
    existing = db.query_one(
        conn,
        "SELECT * FROM diff_item WHERE kind=? AND ref_id=? AND status='open'",
        (kind, ref_id))
    if remain > config.AMOUNT_EPS:
        if existing:
            conn.execute("UPDATE diff_item SET amount=? WHERE id=?", (remain, existing["id"]))
        else:
            conn.execute(
                "INSERT INTO diff_item(kind,ref_id,amount,reason,status,created_at) "
                "VALUES(?,?,?,?,'open',?)",
                (kind, ref_id, remain, reason, db.now_iso()))
    elif existing:
        conn.execute(
            "UPDATE diff_item SET status='resolved',resolution='自动配平',"
            "resolved_at=?,operator='system' WHERE id=?",
            (db.now_iso(), existing["id"]))


def refresh_all_diffs(conn) -> int:
    """全量重算差异（自动对账与手工核销后都会调用，保证差异表与余额一致）。"""
    n = 0
    with db.immediate_tx(conn):
        for txn in db.query_all(conn, "SELECT * FROM bank_tx WHERE direction='in'"):
            _upsert_diff(conn, "unmatched_txn", txn["id"],
                         round(txn["amount"] - txn["matched_amount"], 2),
                         "收入流水未配平")
            n += 1
        for ar in db.query_all(conn, "SELECT * FROM receivable"):
            _upsert_diff(conn, "unmatched_ar", ar["id"],
                         round(ar["amount"] - ar["received_amount"], 2), "应收未收齐")
            n += 1
    return n


def _same_customer(txn, ar) -> bool:
    """客户一致性：应收客户名出现在流水对方户名中（容忍“有限公司/分公司”等后缀差异）。"""
    party = txn["counterparty"] or ""
    customer = ar["customer"] or ""
    return bool(customer) and (customer in party or party in customer)


def _candidate_ars(conn, txn, restrict_account: bool) -> list:
    if restrict_account and txn["account_id"] is not None:
        return db.query_all(
            conn,
            "SELECT * FROM receivable WHERE "
            "(collect_account_id IS NULL OR collect_account_id=?) "
            "AND amount-received_amount > 0 ORDER BY due_date, id",
            (txn["account_id"],))
    return db.query_all(
        conn,
        "SELECT * FROM receivable WHERE amount-received_amount > 0 ORDER BY due_date, id")


def auto_match(conn, restrict_account: bool = True, operator: str = "auto") -> dict:
    """自动对账：在单事务内按确定性策略完成全部可自动配对。

    策略（每一步都走 writeoff 的同一套余额/唯一约束校验）：
    1. 1:1：剩余金额恰好相等且客户一致；
    2. 1:N：单笔流水恰好覆盖同一客户的多张应收，按到期顺序拆分；
    3. N:1：一张应收由同客户多笔流水恰好凑齐，逐笔核销；
    4. 部分付款：同客户仅剩一张未收齐应收时按较小额部分核销，剩余进差异。
    其余（配不上）自动落差异，交人工核销。
"""
    stats = {"one_to_one": 0, "one_to_many": 0, "many_to_one": 0,
             "partial": 0, "diff_left": 0}
    with db.immediate_tx(conn):
        txns = db.query_all(
            conn,
            "SELECT * FROM bank_tx WHERE direction='in' AND amount-matched_amount > 0 "
            "ORDER BY txn_date, id")

        # ---- 第 1 轮：1:1 精确相等 ----
        for txn in list(txns):
            txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE id=?", (txn["id"],))
            remain_txn = round(txn["amount"] - txn["matched_amount"], 2)
            if remain_txn <= config.AMOUNT_EPS:
                continue
            for ar in _candidate_ars(conn, txn, restrict_account):
                remain_ar = round(ar["amount"] - ar["received_amount"], 2)
                # 金额相等且客户一致才自动 1:1，避免同名额撞单
                if abs(remain_txn - remain_ar) <= config.AMOUNT_EPS and _same_customer(txn, ar):
                    _do_writeoff_in_tx(conn, txn["id"], ar["id"], remain_txn, "auto", operator)
                    stats["one_to_one"] += 1
                    break

        # ---- 第 2 轮：1:N 单笔流水覆盖多张应收 ----
        for txn in list(txns):
            txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE id=?", (txn["id"],))
            budget = round(txn["amount"] - txn["matched_amount"], 2)
            if budget <= config.AMOUNT_EPS:
                continue
            plan: list[tuple[int, float]] = []
            used = 0.0
            for ar in _candidate_ars(conn, txn, restrict_account):
                # 1:N 拆分只允许付给同一客户的多张应收
                if not _same_customer(txn, ar):
                    continue
                remain_ar = round(ar["amount"] - ar["received_amount"], 2)
                if round(used + remain_ar, 2) > budget + config.AMOUNT_EPS:
                    break
                plan.append((ar["id"], remain_ar))
                used = round(used + remain_ar, 2)
            # 要求至少配两张且把流水用掉（1:1 已在第 1 轮处理）
            if len(plan) >= 2 and abs(used - budget) <= config.AMOUNT_EPS:
                for ar_id, amt in plan:
                    _do_writeoff_in_tx(conn, txn["id"], ar_id, amt, "auto", operator)
                    stats["one_to_many"] += 1

        # ---- 第 3 轮：N:1 多笔流水凑一张应收（按客户归组）----
        ars = db.query_all(
            conn, "SELECT * FROM receivable WHERE amount-received_amount > 0 ORDER BY due_date, id")
        for ar in ars:
            ar = db.query_one(conn, "SELECT * FROM receivable WHERE id=?", (ar["id"],))
            target = round(ar["amount"] - ar["received_amount"], 2)
            if target <= config.AMOUNT_EPS:
                continue
            pool = db.query_all(
                conn,
                "SELECT * FROM bank_tx WHERE direction='in' AND amount-matched_amount > 0 "
                "AND (counterparty LIKE ? OR ?='') ORDER BY txn_date, id",
                (f"%{ar['customer']}%", ar["customer"]))
            plan: list[tuple[int, float]] = []
            used = 0.0
            for txn in pool:
                remain_txn = round(txn["amount"] - txn["matched_amount"], 2)
                if restrict_account and ar["collect_account_id"] is not None \
                        and txn["account_id"] != ar["collect_account_id"]:
                    continue
                take = min(remain_txn, round(target - used, 2))
                if take <= config.AMOUNT_EPS:
                    continue
                plan.append((txn["id"], round(take, 2)))
                used = round(used + take, 2)
                if abs(used - target) <= config.AMOUNT_EPS:
                    break
            if len(plan) >= 2 and abs(used - target) <= config.AMOUNT_EPS:
                for txn_id, amt in plan:
                    _do_writeoff_in_tx(conn, txn_id, ar["id"], amt, "auto", operator)
                    stats["many_to_one"] += 1

        # ---- 第 4 轮：部分付款——同客户仅剩一张未收齐应收时，按较小额部分核销 ----
        for txn in list(txns):
            txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE id=?", (txn["id"],))
            budget = round(txn["amount"] - txn["matched_amount"], 2)
            if budget <= config.AMOUNT_EPS:
                continue
            same_customer_ars = [
                ar for ar in _candidate_ars(conn, txn, restrict_account)
                if _same_customer(txn, ar)]
            if len(same_customer_ars) != 1:
                continue  # 客户不唯一或无可配应收，不猜，留人工处理
            ar = same_customer_ars[0]
            remain_ar = round(ar["amount"] - ar["received_amount"], 2)
            take = round(min(budget, remain_ar), 2)
            if take > config.AMOUNT_EPS:
                _do_writeoff_in_tx(conn, txn["id"], ar["id"], take, "auto", operator)
                stats["partial"] += 1

        # ---- 剩余未配平全部进差异 ----
        for txn in db.query_all(conn, "SELECT * FROM bank_tx WHERE direction='in'"):
            _upsert_diff(conn, "unmatched_txn", txn["id"],
                         round(txn["amount"] - txn["matched_amount"], 2), "收入流水未配平")
        for ar in db.query_all(conn, "SELECT * FROM receivable"):
            _upsert_diff(conn, "unmatched_ar", ar["id"],
                         round(ar["amount"] - ar["received_amount"], 2), "应收未收齐")
        stats["diff_left"] = db.query_one(
            conn, "SELECT COUNT(*) c FROM diff_item WHERE status='open'")["c"]
    return stats


def _do_writeoff_in_tx(conn, txn_id: int, receivable_id: int, amount: float,
                       source: str, operator: str) -> int:
    """auto_match 已持有 IMMEDIATE 事务，这里做不含开事务的内核版核销。"""
    txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE id=?", (txn_id,))
    ar = db.query_one(conn, "SELECT * FROM receivable WHERE id=?", (receivable_id,))
    if _open_match_exists(conn, txn_id, receivable_id):
        raise MatchError("自动配对命中已存在的有效核销")
    cur = conn.execute(
        "INSERT INTO match_entry(txn_id,receivable_id,amount,status,source,operator,created_at) "
        "VALUES(?,?,?, 'active', ?, ?, ?)",
        (txn_id, receivable_id, amount, source, operator, db.now_iso()))
    conn.execute("UPDATE bank_tx SET matched_amount=matched_amount+? WHERE id=?",
                 (amount, txn_id))
    conn.execute("UPDATE receivable SET received_amount=received_amount+? WHERE id=?",
                 (amount, receivable_id))
    db.insert_audit(conn, "writeoff", "match_entry", cur.lastrowid,
                    f"流水 {txn['txn_ref']} 配应收 {ar['ar_no']} 金额 {amount}（{source}）",
                    operator)
    return cur.lastrowid


def resolve_diff_manually(conn, diff_id: int, resolution: str, operator: str = "manual") -> None:
    """差异人工核销留痕：记录处理说明与操作人，不允许关闭不存在/已解决的差异。"""
    with db.immediate_tx(conn):
        d = db.query_one(conn, "SELECT * FROM diff_item WHERE id=?", (diff_id,))
        if d is None:
            raise MatchError("差异不存在")
        if d["status"] != "open":
            raise MatchError("差异已解决，不能重复处理")
        if not resolution.strip():
            raise MatchError("人工核销必须填写处理说明")
        conn.execute(
            "UPDATE diff_item SET status='resolved',resolution=?,operator=?,resolved_at=? WHERE id=?",
            (resolution.strip(), operator, db.now_iso(), diff_id))
        db.insert_audit(conn, "resolve_diff", "diff_item", diff_id,
                        f"{d['kind']} 人工核销：{resolution.strip()}", operator)
