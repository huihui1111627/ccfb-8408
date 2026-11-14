"""解析层与 PDF 回单抽取的单元测试。"""
from __future__ import annotations

import pytest

from bankrpa.rpa.config_loader import get_bank_config
from bankrpa.rpa.parser import normalize_row, parse_amount, parse_direction, split_txn_date
from bankrpa.rpa.receipt import extract_receipt_fields
from bankrpa.mock_bank.data import BANKS
from bankrpa.mock_bank.pdf_receipt import build_receipt_pdf


def test_parse_amount_variants():
    assert parse_amount("￥12,000.50 元") == 12000.50
    assert parse_amount("3,500") == 3500.0
    assert parse_amount("") == 0.0
    assert parse_amount(None) == 0.0
    assert parse_amount("-128.00") == -128.0


def test_parse_direction():
    assert parse_direction("收入", ["收入", "贷"], ["支出", "借"]) == "in"
    assert parse_direction("借", ["收入", "贷"], ["支出", "借"]) == "out"
    with pytest.raises(ValueError):
        parse_direction("转账", ["收入"], ["支出"])


def test_split_txn_date():
    assert split_txn_date("2026-11-13 09:12:21") == "2026-11-13"
    assert split_txn_date("2026/11/12") == "2026-11-12"
    assert split_txn_date("2026-11-13") == "2026-11-13"


def test_normalize_split_mode_scb():
    cfg = get_bank_config("scb")
    raw = {"txn_time": "2026-11-13 08:58:33", "txn_ref": "SCB-X1",
           "credit": "3,500.00", "debit": "", "counterparty": "甲",
           "summary": "货款", "receipt": "/receipt/SCB-X1"}
    out = normalize_row(raw, cfg)
    assert out["direction"] == "in" and out["amount"] == 3500.0
    assert out["txn_date"] == "2026-11-13" and out["txn_ref"] == "SCB-X1"


def test_normalize_signed_mode_ecd():
    cfg = get_bank_config("ecd")
    raw = {"txn_time": "2026-11-12 14:30:02", "txn_ref": "ECD-X9",
           "direction": "支出", "amount": "3000.00", "counterparty": "乙",
           "summary": "广告费", "receipt": ""}
    out = normalize_row(raw, cfg)
    assert out["direction"] == "out" and out["amount"] == 3000.0


def test_split_mode_both_columns_nonzero_rejected():
    cfg = get_bank_config("scb")
    raw = {"txn_time": "2026-11-13 08:58:33", "txn_ref": "X",
           "credit": "100.00", "debit": "200.00", "counterparty": "",
           "summary": "", "receipt": ""}
    with pytest.raises(ValueError):
        normalize_row(raw, cfg)


def test_receipt_pdf_extraction():
    txn = BANKS["ecd"]["txns"][0]
    pdf = build_receipt_pdf("华东发展银行", txn)
    fields = extract_receipt_fields(pdf)
    assert fields["payer"] == txn["payer"]
    assert fields["purpose"] == txn["purpose"]
    assert txn["ref"] in fields["raw_text"]
