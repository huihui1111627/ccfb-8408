"""网银对账 RPA 网页控制台（FastAPI + Jinja2 服务端渲染）。

页面：银行账户 / 下载任务 / 流水库 / 应收台账 / 自动对账 / 差异处理 / 余额调节表 / 操作日志。
后台下载任务走线程池，每个任务在自己的数据库连接里运行；
任务状态由 download_task 表守卫，running 状态拒绝重复执行。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .. import config, db
from ..services import reconcile, report as report_svc, tasks
from .utils import fix_form_utf8

TEMPLATES_DIR = config.PACKAGE_DIR / "web" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="rpa-task")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="网银对账 RPA 控制台", lifespan=lifespan)


def _conn():
    return db.connect()


def _render(request: Request, template: str, **ctx) -> HTMLResponse:
    ctx.setdefault("today", date.today().isoformat())
    return templates.TemplateResponse(request, template, ctx)


# ---------------- 首页 / 银行账户 ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    conn = _conn()
    try:
        accounts = db.query_all(conn, """
            SELECT a.*,
              (SELECT COUNT(*) FROM bank_tx t WHERE t.account_id=a.id) AS txn_count,
              (SELECT COUNT(*) FROM download_task k WHERE k.account_id=a.id
                 AND k.status='success') AS task_count
            FROM bank_account a ORDER BY a.id""")
        return _render(request, "accounts.html", accounts=accounts)
    finally:
        conn.close()


# ---------------- 下载任务 ----------------
@app.get("/tasks", response_class=HTMLResponse)
def task_list(request: Request, account_id: int | None = None):
    conn = _conn()
    try:
        accounts = db.query_all(conn, "SELECT * FROM bank_account ORDER BY id")
        sql = ("SELECT k.*, a.bank_name, a.account_no FROM download_task k "
               "JOIN bank_account a ON a.id=k.account_id")
        params: list = []
        if account_id:
            sql += " WHERE k.account_id=?"
            params.append(account_id)
        sql += " ORDER BY k.id DESC LIMIT 200"
        rows = db.query_all(conn, sql, params)
        return _render(request, "tasks.html", tasks=rows, accounts=accounts,
                       selected=account_id)
    finally:
        conn.close()


@app.post("/tasks/create")
def task_create(request: Request, account_id: int = Form(...),
                date_from: str = Form(...), date_to: str = Form(...)):
    conn = _conn()
    try:
        tasks.create_task(conn, account_id, date_from, date_to)
    finally:
        conn.close()
    return RedirectResponse("/tasks", status_code=303)


@app.post("/tasks/incremental/{account_id}")
def task_incremental(request: Request, account_id: int):
    conn = _conn()
    try:
        task_id = tasks.create_incremental_task(conn, account_id)
    finally:
        conn.close()
    _submit(task_id)
    return RedirectResponse("/tasks", status_code=303)


@app.post("/tasks/run/{task_id}")
def task_run(request: Request, task_id: int):
    _submit(task_id)
    return RedirectResponse("/tasks", status_code=303)


def _submit(task_id: int) -> None:
    def _job():
        conn = db.connect()
        try:
            tasks.run_task(conn, task_id, headless=True)
        except Exception as exc:  # 失败已落库，这里仅避免线程异常外泄
            print(f"[task {task_id}] failed: {exc}")
        finally:
            conn.close()
    _executor.submit(_job)


# ---------------- 流水库 ----------------
@app.get("/txns", response_class=HTMLResponse)
def txn_list(request: Request, account_id: int | None = None,
             only_unmatched: int = 0, q: str = ""):
    conn = _conn()
    try:
        accounts = db.query_all(conn, "SELECT * FROM bank_account ORDER BY id")
        sql = "SELECT t.*, a.bank_name FROM bank_tx t JOIN bank_account a ON a.id=t.account_id WHERE 1=1"
        params: list = []
        if account_id:
            sql += " AND t.account_id=?"
            params.append(account_id)
        if only_unmatched:
            sql += " AND t.amount-t.matched_amount > ?"
            params.append(config.AMOUNT_EPS)
        if q:
            like = f"%{q}%"
            sql += " AND (t.txn_ref LIKE ? OR t.counterparty LIKE ? OR t.summary LIKE ?)"
            params.extend([like, like, like])
        sql += " ORDER BY t.txn_date DESC, t.id DESC LIMIT 500"
        rows = db.query_all(conn, sql, params)
        return _render(request, "txns.html", txns=rows, accounts=accounts,
                       selected=account_id, only_unmatched=only_unmatched, q=q)
    finally:
        conn.close()


@app.get("/txns/receipt/{txn_id}")
def txn_receipt(request: Request, txn_id: int):
    conn = _conn()
    try:
        row = db.query_one(conn, "SELECT receipt_path FROM bank_tx WHERE id=?", (txn_id,))
        if not row or not row["receipt_path"]:
            return Response("回单不存在", status_code=404)
        with open(row["receipt_path"], "rb") as f:
            data = f.read()
        return Response(data, media_type="application/pdf",
                        headers={"Content-Disposition": 'inline; filename="receipt.pdf"'})
    finally:
        conn.close()


# ---------------- 应收台账 ----------------
@app.get("/receivables", response_class=HTMLResponse)
def ar_list(request: Request):
    conn = _conn()
    try:
        rows = db.query_all(
            conn,
            "SELECT r.*, a.bank_name FROM receivable r LEFT JOIN bank_account a "
            "ON a.id=r.collect_account_id ORDER BY r.due_date, r.id")
        accounts = db.query_all(conn, "SELECT * FROM bank_account ORDER BY id")
        return _render(request, "receivables.html", receivables=rows, accounts=accounts)
    finally:
        conn.close()


@app.post("/receivables/create")
async def ar_create(request: Request):
    form = fix_form_utf8(await request.form())
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO receivable(ar_no,customer,amount,received_amount,"
            "collect_account_id,due_date,created_at) VALUES(?,?,?,0,?,?,?)",
            (form["ar_no"], form["customer"], float(form["amount"]),
             form.get("collect_account_id") or None, form["due_date"], db.now_iso()))
        db.insert_audit(conn, "create_receivable", "receivable", None,
                        f"{form['ar_no']} {form['customer']} {form['amount']}")
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse("/receivables", status_code=303)


# ---------------- 自动对账 + 手工核销 ----------------
@app.get("/reconcile", response_class=HTMLResponse)
def reconcile_page(request: Request):
    conn = _conn()
    try:
        open_txn = db.query_all(
            conn,
            "SELECT t.*, a.bank_name FROM bank_tx t JOIN bank_account a ON a.id=t.account_id "
            "WHERE t.direction='in' AND t.amount-t.matched_amount > ? ORDER BY t.txn_date, t.id",
            (config.AMOUNT_EPS,))
        open_ar = db.query_all(
            conn,
            "SELECT r.*, a.bank_name FROM receivable r LEFT JOIN bank_account a "
            "ON a.id=r.collect_account_id WHERE r.amount-r.received_amount > ? "
            "ORDER BY r.due_date, r.id", (config.AMOUNT_EPS,))
        entries = db.query_all(
            conn,
            "SELECT m.*, t.txn_ref, t.amount AS txn_amount, r.ar_no, r.customer "
            "FROM match_entry m JOIN bank_tx t ON t.id=m.txn_id "
            "JOIN receivable r ON r.id=m.receivable_id "
            "WHERE m.status='active' ORDER BY m.id DESC LIMIT 200")
        return _render(request, "reconcile.html", open_txn=open_txn, open_ar=open_ar,
                       entries=entries)
    finally:
        conn.close()


@app.post("/reconcile/auto")
def reconcile_auto(request: Request):
    conn = _conn()
    try:
        stats = reconcile.auto_match(conn)
        db.insert_audit(conn, "auto_match", "match_entry", None, str(stats))
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse("/reconcile", status_code=303)


@app.post("/reconcile/manual")
async def reconcile_manual(request: Request):
    form = fix_form_utf8(await request.form())
    conn = _conn()
    try:
        reconcile.writeoff(conn, int(form["txn_id"]), int(form["receivable_id"]),
                           float(form["amount"]), source="manual",
                           operator=form.get("operator") or "manual")
        conn.commit()
    except reconcile.MatchError as exc:
        return _error_page(request, f"手工核销失败：{exc}")
    finally:
        conn.close()
    return RedirectResponse("/reconcile", status_code=303)


@app.post("/reconcile/reverse/{match_id}")
def reconcile_reverse(request: Request, match_id: int):
    conn = _conn()
    try:
        reconcile.reverse_writeoff(conn, match_id)
        conn.commit()
    except reconcile.MatchError as exc:
        return _error_page(request, f"冲销失败：{exc}")
    finally:
        conn.close()
    return RedirectResponse("/reconcile", status_code=303)


# ---------------- 差异处理 ----------------
@app.get("/diffs", response_class=HTMLResponse)
def diff_list(request: Request, status_filter: str = "open"):
    conn = _conn()
    try:
        rows = db.query_all(
            conn,
            """SELECT d.*,
                 CASE d.kind WHEN 'unmatched_txn'
                   THEN (SELECT txn_ref FROM bank_tx WHERE id=d.ref_id)
                   ELSE (SELECT ar_no FROM receivable WHERE id=d.ref_id) END AS ref_no,
                 CASE d.kind WHEN 'unmatched_txn'
                   THEN (SELECT counterparty FROM bank_tx WHERE id=d.ref_id)
                   ELSE (SELECT customer FROM receivable WHERE id=d.ref_id) END AS ref_name
               FROM diff_item d
               WHERE (?='all' OR d.status=?) ORDER BY d.status DESC, d.id DESC""",
            (status_filter, status_filter))
        return _render(request, "diffs.html", diffs=rows, status_filter=status_filter)
    finally:
        conn.close()


@app.post("/diffs/resolve/{diff_id}")
async def diff_resolve(request: Request, diff_id: int):
    form = fix_form_utf8(await request.form())
    conn = _conn()
    try:
        reconcile.resolve_diff_manually(conn, diff_id, form.get("resolution", ""),
                                        operator=form.get("operator") or "manual")
        conn.commit()
    except reconcile.MatchError as exc:
        return _error_page(request, f"差异处理失败：{exc}")
    finally:
        conn.close()
    return RedirectResponse("/diffs", status_code=303)


# ---------------- 余额调节表 ----------------
@app.get("/report", response_class=HTMLResponse)
def report_page(request: Request, account_id: int | None = None, report_date: str | None = None):
    conn = _conn()
    try:
        accounts = db.query_all(conn, "SELECT * FROM bank_account ORDER BY id")
        report_date = report_date or date.today().isoformat()
        reports = []
        targets = [account_id] if account_id else [a["id"] for a in accounts]
        for acct_id in targets:
            reports.append(report_svc.build_report(conn, acct_id, report_date))
        return _render(request, "report.html", reports=reports, accounts=accounts,
                       selected=account_id, report_date=report_date)
    finally:
        conn.close()


# ---------------- 操作日志 ----------------
@app.get("/audit", response_class=HTMLResponse)
def audit_list(request: Request):
    conn = _conn()
    try:
        rows = db.query_all(conn, "SELECT * FROM audit_log ORDER BY id DESC LIMIT 300")
        return _render(request, "audit.html", logs=rows)
    finally:
        conn.close()


def _error_page(request: Request, message: str) -> HTMLResponse:
    return _render(request, "error.html", message=message)
