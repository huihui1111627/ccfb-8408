"""Web 控制台 HTTP 层测试（TestClient，不另起端口）。"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from bankrpa import config, db
from bankrpa.web.app import app
from bankrpa.services import tasks


@pytest.fixture()
def client(conn, db_path, monkeypatch):
    # app 内 _conn 走默认 config.DB_PATH，重定向到本测试的临时库
    monkeypatch.setattr(config, "DB_PATH", db_path)
    with TestClient(app) as c:
        yield c


def test_pages_render(client):
    for path in ("/", "/tasks", "/txns", "/receivables", "/reconcile",
                 "/diffs", "/report", "/audit"):
        resp = client.get(path)
        assert resp.status_code == 200, path


def test_create_receivable_chinese_utf8(client, conn):
    resp = client.post("/receivables/create", data={
        "ar_no": "R999", "customer": "北京测试客户有限公司",
        "amount": "1234.56", "collect_account_id": "", "due_date": "2026-12-01",
    }, follow_redirects=False)
    assert resp.status_code == 303
    row = db.query_one(conn, "SELECT * FROM receivable WHERE ar_no='R999'")
    assert row is not None
    assert row["customer"] == "北京测试客户有限公司"  # 无 latin-1 mojibake


def test_task_lifecycle_and_txn_page(client, conn, mock_servers, account_id):
    resp = client.post("/tasks/create", data={
        "account_id": str(account_id("ecd")),
        "date_from": "2026-11-13", "date_to": "2026-11-13",
    }, follow_redirects=False)
    assert resp.status_code == 303
    tid = db.query_one(conn, "SELECT MAX(id) m FROM download_task")["m"]
    # 同步直接执行（绕过线程池，测试里更可控）
    tasks.run_task(conn, tid, headless=True)
    task = db.query_one(conn, "SELECT * FROM download_task WHERE id=?", (tid,))
    assert task["status"] == "success" and task["rows_inserted"] >= 3

    resp = client.get("/txns")
    assert resp.status_code == 200 and "ECD-20261113".encode() in resp.content

    # 回单 PDF 可下载
    txn_id = db.query_one(conn, "SELECT id FROM bank_tx LIMIT 1")["id"]
    resp = client.get(f"/txns/receipt/{txn_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"


def test_manual_writeoff_and_diff_flow_via_http(client, conn, mock_servers, account_id):
    for code in ("ecd",):
        tid = tasks.create_task(conn, account_id(code), "2026-11-12", "2026-11-13")
        tasks.run_task(conn, tid, headless=True)
    txn = db.query_one(conn, "SELECT * FROM bank_tx WHERE txn_ref='ECD-20261113-001'")
    ar = db.query_one(conn, "SELECT * FROM receivable WHERE ar_no='R001'")
    resp = client.post("/reconcile/manual", data={
        "txn_id": str(txn["id"]), "receivable_id": str(ar["id"]),
        "amount": "10000.00", "operator": "web测试员",
    }, follow_redirects=False)
    assert resp.status_code == 303
    assert db.query_one(
        conn, "SELECT matched_amount FROM bank_tx WHERE id=?", (txn["id"],)
    )["matched_amount"] == 10000.0

    # 重复核销返回错误页而不是 500（错误页不经重定向，直接 200 渲染）
    resp = client.post("/reconcile/manual", data={
        "txn_id": str(txn["id"]), "receivable_id": str(ar["id"]),
        "amount": "10000.00", "operator": "web测试员",
    }, follow_redirects=False)
    assert "不能重复核销".encode() in resp.content


def test_auto_and_diff_resolve_via_http(client, conn, mock_servers, account_id):
    for code in ("ecd", "scb", "wcb"):
        tid = tasks.create_task(conn, account_id(code), "2026-11-12", "2026-11-13")
        tasks.run_task(conn, tid, headless=True)
    assert client.post("/reconcile/auto", follow_redirects=False).status_code == 303
    diff = db.query_one(conn, "SELECT * FROM diff_item WHERE status='open' LIMIT 1")
    assert diff is not None
    resp = client.post(f"/diffs/resolve/{diff['id']}", data={
        "resolution": "控制台人工核销说明", "operator": "web测试员",
    }, follow_redirects=False)
    assert resp.status_code == 303
    assert db.query_one(conn, "SELECT status FROM diff_item WHERE id=?",
                        (diff["id"],))["status"] == "resolved"


def test_report_page_with_date(client):
    resp = client.get("/report?report_date=2026-11-13")
    assert resp.status_code == 200
    assert "余额调节表".encode() in resp.content
