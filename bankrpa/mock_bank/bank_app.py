"""三家 DOM 刻意不同的 mock 网银。

- ecd（华东发展银行）：input/table 用 id，翻页是 a#nextPage
- scb（南方商业银行）：class 风格 + table.transaction-grid，next 按钮带 class
- wcb（西部普惠银行）：data-* 属性风格 + ul/li 卡片列表，下一页用 button[data-act=next]

差异全部体现在 bank_configs.yaml，RPA 引擎零硬编码选择器。
"""
from __future__ import annotations

import secrets as _secrets

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import data as mock_data
from .pdf_receipt import build_receipt_pdf
from .token_service import verify_code

SESSIONS: dict[str, str] = {}  # token -> bank_code


def _new_session(bank_code: str) -> str:
    token = _secrets.token_hex(16)
    SESSIONS[token] = bank_code
    return token


def _bank_of(request: Request) -> str | None:
    token = request.cookies.get("sid")
    return SESSIONS.get(token) if token else None


def _login_page(bank: dict, error: str = "") -> str:
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{bank['bank_name']} 网银登录</title></head><body>
<h2>{bank['bank_name']} 企业网银</h2>
{ f'<p style="color:red">{error}</p>' if error else '' }
<form method="post" action="/login">
  <label>用户名 <input name="username" id="loginUser" class="user-input" data-field="user"></label><br>
  <label>密码 <input name="password" type="password" id="loginPwd" class="pwd-input" data-field="pwd"></label><br>
  <label>动态口令 <input name="otp" id="dynamicOtp" class="otp-input" data-field="otp"></label><br>
  <button type="submit" id="loginBtn" class="submit-btn" data-act="login">登录</button>
</form></body></html>"""


# ---------- ecd：id 风格 + 传统 table，a#nextPage ----------
def _ecd_page(bank_code: str, bank: dict, rows: list[dict], page: int, pages: int,
              date_from: str, date_to: str) -> str:
    trs = "".join(
        f"""<tr>
<td>{t['time']}</td><td>{t['ref']}</td>
<td>{'收入' if t['direction']=='in' else '支出'}</td>
<td>{t['amount']:.2f}</td><td>{t['counterparty']}</td><td>{t['summary']}</td>
<td><a href="/receipt/{t['ref']}">查看回单</a></td>
</tr>""" for t in rows)
    next_attr = 'href="#" id="nextPage"' if page < pages else 'href="#" id="nextPage" class="disabled"'
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{bank['bank_name']} 交易明细</title></head><body>
<h2>{bank['bank_name']} | 账号 {bank['account_no']} | 当前余额 <span id="balance">{bank['balance']:.2f}</span></h2>
<form method="get" action="/transactions">
  <input name="from" id="dateFrom" value="{date_from}">
  <input name="to" id="dateTo" value="{date_to}">
  <button type="submit" id="queryBtn">查询</button>
</form>
<table id="txnTable" border="1">
<thead><tr><th>交易时间</th><th>流水号</th><th>借贷方向</th><th>交易金额</th><th>对方户名</th><th>摘要</th><th>回单</th></tr></thead>
<tbody>{trs}</tbody></table>
<div>第 <span id="pageNo">{page}</span> / {pages} 页
<a {next_attr}>下一页</a></div>
<script>
document.getElementById('nextPage').addEventListener('click', function(e){{
  e.preventDefault();
  if (this.classList.contains('disabled')) return;
  const u = new URL(window.location.href);
  u.searchParams.set('page', {page + 1});
  window.location.href = u.toString();
}});
</script>
</body></html>"""


# ---------- scb：class 风格 + 收入/支出两列金额，button.btn-next ----------
def _scb_page(bank_code: str, bank: dict, rows: list[dict], page: int, pages: int,
              date_from: str, date_to: str) -> str:
    trs = "".join(
        f"""<tr>
<td class="col-time">{t['time']}</td><td class="col-ref">{t['ref']}</td>
<td class="col-in">{f"{t['amount']:.2f}" if t['direction']=='in' else ''}</td>
<td class="col-out">{f"{t['amount']:.2f}" if t['direction']=='out' else ''}</td>
<td class="col-party">{t['counterparty']}</td><td class="col-summary">{t['summary']}</td>
<td class="col-receipt"><a class="receipt-link" href="/receipt/{t['ref']}">电子回单</a></td>
</tr>""" for t in rows)
    disabled = " disabled" if page >= pages else ""
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{bank['bank_name']} 明细查询</title></head><body>
<div class="bank-header"><h2>{bank['bank_name']}</h2>
<div class="account-meta">账号 {bank['account_no']} 余额 <span class="balance-value">{bank['balance']:.2f}</span></div></div>
<form class="filter-form" method="get" action="/transactions">
  <input class="date-range start" name="from" value="{date_from}">
  <input class="date-range end" name="to" value="{date_to}">
  <button class="query-button" type="submit">查询明细</button>
</form>
<table class="transaction-grid" border="1">
<thead><tr><th>交易时间</th><th>流水号</th><th>收入金额</th><th>支出金额</th><th>对方户名</th><th>摘要</th><th>回单</th></tr></thead>
<tbody>{trs}</tbody></table>
<div class="pager"><span class="page-info">{page}/{pages}</span>
<button class="btn-next" id="goNext"{disabled}>下一页</button></div>
<script>
const btn = document.querySelector('.btn-next');
if (btn) btn.addEventListener('click', function(){{
  if (this.disabled) return;
  const u = new URL(window.location.href);
  u.searchParams.set('page', {page + 1});
  window.location.href = u.toString();
}});
</script>
</body></html>"""


# ---------- wcb：data-* 风格 + 卡片列表，button[data-act=next] ----------
def _wcb_page(bank_code: str, bank: dict, rows: list[dict], page: int, pages: int,
              date_from: str, date_to: str) -> str:
    cards = "".join(
        f"""<li class="txn-card" data-ref="{t['ref']}">
<span data-k="time">{t['time']}</span>
<span data-k="ref">{t['ref']}</span>
<span data-k="direction">{'贷' if t['direction']=='in' else '借'}</span>
<span data-k="credit">{f"{t['amount']:.2f}" if t['direction']=='in' else ''}</span>
<span data-k="debit">{f"{t['amount']:.2f}" if t['direction']=='out' else ''}</span>
<span data-k="party">{t['counterparty']}</span>
<span data-k="memo">{t['summary']}</span>
<a data-k="receipt" href="/receipt/{t['ref']}">回单下载</a>
</li>""" for t in rows)
    disabled = " disabled" if page >= pages else ""
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{bank['bank_name']} 流水中心</title></head><body>
<section class="head"><h2>{bank['bank_name']}</h2>
<p>账户 {bank['account_no']}，可用余额 <b data-balance="1">{bank['balance']:.2f}</b></p></section>
<section class="query-panel">
<form method="get" action="/transactions">
  <input type="date" data-input="from" name="from" value="{date_from}">
  <input type="date" data-input="to" name="to" value="{date_to}">
  <button data-act="search" type="submit">检索</button>
</form></section>
<ul class="txn-list" data-role="txn-list">{cards}</ul>
<div class="pagination" data-role="pager">
<span data-page="{page}">{page} / {pages}</span>
<button data-act="next"{disabled}>下一页</button></div>
<script>
const nb = document.querySelector('.pagination button[data-act="next"]');
if (nb) nb.addEventListener('click', function(){{
  if (this.disabled) return;
  const u = new URL(window.location.href);
  u.searchParams.set('page', {page + 1});
  window.location.href = u.toString();
}});
</script>
</body></html>"""


_PAGE_RENDERERS = {"ecd": _ecd_page, "scb": _scb_page, "wcb": _wcb_page}


def create_app(bank_code: str) -> FastAPI:
    bank = mock_data.BANKS[bank_code]
    app = FastAPI(title=f"mock 网银 - {bank['bank_name']}")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if _bank_of(request):
            return RedirectResponse("/transactions", status_code=302)
        return HTMLResponse(_login_page(bank))

    @app.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...), otp: str = Form(...)):
        if username != bank["login_user"] or password != bank["login_pwd"]:
            return HTMLResponse(_login_page(bank, "用户名或密码错误"), status_code=401)
        if not verify_code(username, otp):
            return HTMLResponse(_login_page(bank, "动态口令错误或已过期"), status_code=401)
        token = _new_session(bank_code)
        resp = RedirectResponse("/transactions", status_code=302)
        resp.set_cookie("sid", token, httponly=True)
        return resp

    @app.get("/transactions", response_class=HTMLResponse)
    def transactions(request: Request, startDate: str | None = None, endDate: str | None = None,
                     page: int = 1):
        # 三家参数名一致，但 DOM 结构完全不同（DOM 差异由 bank_configs.yaml 吸收）
        date_from = startDate or ""
        date_to = endDate or ""
        if _bank_of(request) != bank_code:
            return RedirectResponse("/", status_code=302)
        all_rows = mock_data.filter_txns(bank_code, date_from or None, date_to or None)
        size = bank["page_size"]
        pages = max(1, (len(all_rows) + size - 1) // size)
        page = max(1, min(page, pages))
        rows = all_rows[(page - 1) * size:page * size]
        html = _PAGE_RENDERERS[bank_code](bank_code, bank, rows, page, pages, date_from, date_to)
        return HTMLResponse(html)

    @app.get("/api/balance")
    def api_balance(request: Request):
        if _bank_of(request) != bank_code:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"account_no": bank["account_no"], "balance": bank["balance"]}

    @app.get("/receipt/{ref}")
    def receipt(request: Request, ref: str):
        if _bank_of(request) != bank_code:
            return RedirectResponse("/", status_code=302)
        txn = next((t for t in bank["txns"] if t["ref"] == ref), None)
        if txn is None:
            return Response("not found", status_code=404)
        pdf = build_receipt_pdf(bank["bank_name"], txn)
        headers = {"Content-Disposition": f'inline; filename="receipt_{ref}.pdf"'}
        return Response(pdf, media_type="application/pdf", headers=headers)

    @app.get("/health")
    def health():
        return {"status": "ok", "bank": bank_code}

    return app
