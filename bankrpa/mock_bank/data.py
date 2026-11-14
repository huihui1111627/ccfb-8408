"""三家 mock 网银共享的内存数据。

数据刻意覆盖全部对账场景：
- 一对一、一对多（一笔流水对应多张应收）、多对一（多笔流水凑一张应收）、
  部分付款、完全配不上的流水/应收。
- 交易跨 2026-11-12 / 11-13 两天，用于验证增量下载与跨天补漏。
"""
from __future__ import annotations

# 每家银行：登录信息 + 账户 + 余额 + 流水（分页大小刻意很小，逼出翻页逻辑）
BANKS: dict[str, dict] = {
    "ecd": {
        "bank_name": "华东发展银行",
        "login_user": "ecd_corp",
        "login_pwd": "Ecd@123456",
        "account_no": "6228010100001001",
        "account_name": "集团财务共享中心",
        "balance": 158600.00,
        "page_size": 3,
        "txns": [
            {"ref": "ECD-20261113-001", "time": "2026-11-13 09:12:21", "date": "2026-11-13",
             "direction": "in", "amount": 10000.00, "counterparty": "华东医药集团",
             "summary": "货款 R001", "payer": "华东医药集团", "purpose": "货款 R001"},
            {"ref": "ECD-20261113-002", "time": "2026-11-13 10:05:44", "date": "2026-11-13",
             "direction": "in", "amount": 4800.00, "counterparty": "苏州精密机械厂",
             "summary": "往来款", "payer": "苏州精密机械厂", "purpose": "往来款"},
            {"ref": "ECD-20261112-003", "time": "2026-11-12 14:30:02", "date": "2026-11-12",
             "direction": "out", "amount": 3000.00, "counterparty": "上海广告有限公司",
             "summary": "广告费", "payer": "集团财务共享中心", "purpose": "广告费"},
            {"ref": "ECD-20261113-004", "time": "2026-11-13 16:40:18", "date": "2026-11-13",
             "direction": "in", "amount": 5500.00, "counterparty": "南京贸易有限公司",
             "summary": "部分货款 R004", "payer": "南京贸易有限公司", "purpose": "部分货款 R004"},
        ],
    },
    "scb": {
        "bank_name": "南方商业银行",
        "login_user": "scb_corp",
        "login_pwd": "Scb@123456",
        "account_no": "6217020200002002",
        "account_name": "集团财务共享中心",
        "balance": 92400.00,
        "page_size": 3,
        "txns": [
            {"ref": "SCB-20261113-001", "time": "2026-11-13 08:58:33", "date": "2026-11-13",
             "direction": "in", "amount": 3500.00, "counterparty": "苏州精密机械厂",
             "summary": "货款 R002", "payer": "苏州精密机械厂", "purpose": "货款 R002"},
            {"ref": "SCB-20261113-002", "time": "2026-11-13 11:20:09", "date": "2026-11-13",
             "direction": "in", "amount": 2000.00, "counterparty": "宁波物流有限公司",
             "summary": "运费分期 R005", "payer": "宁波物流有限公司", "purpose": "运费分期 R005"},
            {"ref": "SCB-20261113-003", "time": "2026-11-13 15:02:57", "date": "2026-11-13",
             "direction": "in", "amount": 3000.00, "counterparty": "宁波物流有限公司",
             "summary": "运费尾款 R005", "payer": "宁波物流有限公司", "purpose": "运费尾款 R005"},
            {"ref": "SCB-20261112-004", "time": "2026-11-12 13:15:40", "date": "2026-11-12",
             "direction": "in", "amount": 6000.00, "counterparty": "杭州电子科技有限公司",
             "summary": "采购款 R006", "payer": "杭州电子科技有限公司", "purpose": "采购款 R006"},
            {"ref": "SCB-20261112-005", "time": "2026-11-12 17:22:11", "date": "2026-11-12",
             "direction": "in", "amount": 900.00, "counterparty": "无锡化工原料公司",
             "summary": "保证金", "payer": "无锡化工原料公司", "purpose": "保证金"},
        ],
    },
    "wcb": {
        "bank_name": "西部普惠银行",
        "login_user": "wcb_corp",
        "login_pwd": "Wcb@123456",
        "account_no": "6230030300003003",
        "account_name": "集团财务共享中心",
        "balance": 47750.00,
        "page_size": 2,
        "txns": [
            {"ref": "WCB-20261113-001", "time": "2026-11-13 09:40:00", "date": "2026-11-13",
             "direction": "in", "amount": 2500.00, "counterparty": "合肥建材商行",
             "summary": "货款一期 R007", "payer": "合肥建材商行", "purpose": "货款一期 R007"},
            {"ref": "WCB-20261113-002", "time": "2026-11-13 09:41:12", "date": "2026-11-13",
             "direction": "in", "amount": 1500.00, "counterparty": "合肥建材商行",
             "summary": "货款二期 R007", "payer": "合肥建材商行", "purpose": "货款二期 R007"},
            {"ref": "WCB-20261112-003", "time": "2026-11-12 10:10:25", "date": "2026-11-12",
             "direction": "in", "amount": 7000.00, "counterparty": "南昌实业公司",
             "summary": "合并付款 R008 R009", "payer": "南昌实业公司", "purpose": "合并付款 R008 R009"},
            {"ref": "WCB-20261112-004", "time": "2026-11-12 16:48:50", "date": "2026-11-12",
             "direction": "in", "amount": 2000.00, "counterparty": "长沙食品厂",
             "summary": "预付货款 R010", "payer": "长沙食品厂", "purpose": "预付货款 R010"},
        ],
    },
}

# 应收台账（collect_account 对应银行代码，None 表示不限定收款账户）
RECEIVABLES: list[dict] = [
    {"ar_no": "R001", "customer": "华东医药集团", "amount": 10000.00, "bank": "ecd", "due_date": "2026-11-15"},
    {"ar_no": "R002", "customer": "苏州精密机械厂", "amount": 3500.00, "bank": "scb", "due_date": "2026-11-15"},
    {"ar_no": "R003", "customer": "青岛进出口公司", "amount": 8200.00, "bank": "ecd", "due_date": "2026-11-16"},
    {"ar_no": "R004", "customer": "南京贸易有限公司", "amount": 12000.00, "bank": "ecd", "due_date": "2026-11-20"},
    {"ar_no": "R005", "customer": "宁波物流有限公司", "amount": 5000.00, "bank": "scb", "due_date": "2026-11-18"},
    {"ar_no": "R006", "customer": "杭州电子科技有限公司", "amount": 6000.00, "bank": "scb", "due_date": "2026-11-18"},
    {"ar_no": "R007", "customer": "合肥建材商行", "amount": 4000.00, "bank": "wcb", "due_date": "2026-11-19"},
    {"ar_no": "R008", "customer": "南昌实业公司", "amount": 3000.00, "bank": "wcb", "due_date": "2026-11-19"},
    {"ar_no": "R009", "customer": "南昌实业公司", "amount": 4000.00, "bank": "wcb", "due_date": "2026-11-19"},
    {"ar_no": "R010", "customer": "长沙食品厂", "amount": 3800.00, "bank": "wcb", "due_date": "2026-11-22"},
]

# 企业账面余额：按余额调节表公式反推，使自动对账后调节后两侧必然相等
#   账面余额 = 银行余额 + 企业已收银行未收(未收齐应收) - 银行已收企业未收(未配平收入) + 银行已付企业未付(支出)
BOOK_BALANCES = {"ecd": 171500.00, "scb": 91500.00, "wcb": 49550.00}


def filter_txns(bank_code: str, date_from: str | None, date_to: str | None) -> list[dict]:
    out = []
    for t in BANKS[bank_code]["txns"]:
        if date_from and t["date"] < date_from:
            continue
        if date_to and t["date"] > date_to:
            continue
        out.append(t)
    return out
