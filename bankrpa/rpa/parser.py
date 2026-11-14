"""各家流水的统一解析：金额、方向、日期、文本清洗全部归一到内部字段。"""
from __future__ import annotations

import re
from datetime import datetime

_AMOUNT_RE = re.compile(r"[^\d.\-]")


def parse_amount(text: str) -> float:
    """'12,000.00 元' / '￥3,500.00' -> 12000.0 / 3500.0，空串 -> 0.0。"""
    if text is None:
        return 0.0
    cleaned = _AMOUNT_RE.sub("", str(text).replace(",", ""))
    if cleaned in ("", "-", "."):
        return 0.0
    return float(cleaned)


def parse_direction(text: str, in_keywords: list[str], out_keywords: list[str]) -> str:
    t = (text or "").strip()
    if any(k in t for k in in_keywords):
        return "in"
    if any(k in t for k in out_keywords):
        return "out"
    raise ValueError(f"无法识别借贷方向文本: {t!r}")


def split_txn_date(txn_time: str) -> str:
    """'2026-11-13 09:12:21' -> '2026-11-13'；纯日期原样返回。"""
    t = txn_time.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    # 兜底：取前 10 个字符里的日期
    m = re.match(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", t)
    if not m:
        raise ValueError(f"无法解析交易时间: {txn_time!r}")
    return m.group(1).replace("/", "-")


def normalize_row(raw: dict, cfg: dict) -> dict:
    """把按配置抓下来的一行原始单元格，归一成引擎标准流水字典。"""
    amount_cfg = cfg["txn"]["amount"]
    if amount_cfg["mode"] == "split":
        credit = parse_amount(raw.get(amount_cfg["credit_cell"], ""))
        debit = parse_amount(raw.get(amount_cfg["debit_cell"], ""))
        if credit > 0 and debit > 0:
            raise ValueError(f"同一行收入/支出两列都有值，数据异常: {raw}")
        amount = credit + debit
        direction = "in" if credit > 0 else "out"
    elif amount_cfg["mode"] == "signed":
        amount = parse_amount(raw.get(amount_cfg["amount_cell"], ""))
        direction = parse_direction(
            raw.get(amount_cfg["direction_cell"], ""),
            amount_cfg.get("in_keywords", ["收入", "贷"]),
            amount_cfg.get("out_keywords", ["支出", "借"]),
        )
    else:
        raise ValueError(f"未知金额解析模式: {amount_cfg['mode']}")

    txn_time = (raw.get("txn_time") or "").strip()
    return {
        "txn_ref": (raw.get("txn_ref") or "").strip(),
        "txn_time": txn_time,
        "txn_date": split_txn_date(txn_time),
        "direction": direction,
        "amount": round(amount, 2),
        "counterparty": (raw.get("counterparty") or "").strip(),
        "summary": (raw.get("summary") or "").strip(),
        "receipt_href": (raw.get("receipt") or "").strip(),
    }
