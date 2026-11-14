"""配置驱动的 Playwright 下载引擎。

本模块不允许出现任何银行代码/硬编码选择器——所有步骤来自 bank_configs.yaml。
防呆设计：
- 翻页用“页内行指纹”去重，指纹重复即判定翻页失效，避免死循环；
- next 选择器支持候选列表，第一个命中的生效；
- 每个动作有显式等待与超时，失败带上下文抛错，由任务层落库。
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

from .. import config
from .parser import normalize_row
from .receipt import extract_receipt_fields


class RpaError(RuntimeError):
    pass


def fetch_otp(token_url: str, username: str, timeout: float = 10.0) -> str:
    """从 mock 令牌接口取当前动态口令。"""
    sep = "&" if "?" in token_url else "?"
    url = token_url.replace("{username}", urllib.parse.quote(username))
    if "{username}" not in token_url:
        url = f"{token_url}{sep}account={urllib.parse.quote(username)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    otp = payload.get("otp")
    if not otp:
        raise RpaError(f"令牌接口未返回 otp: {payload}")
    return str(otp)


def _resolve(selector: str):
    """配置里统一 'css:xxx' 前缀；未来可扩展 xpath:/text: 等。"""
    if selector.startswith("css:"):
        return selector[4:]
    return selector


def _first_visible(page, candidates: list[str], timeout_ms: int):
    """在候选选择器里找一个当前可见的元素，都不可见返回 (None, None)。"""
    deadline_step = max(200, timeout_ms // max(4, len(candidates)))
    for sel in candidates:
        css = _resolve(sel)
        try:
            el = page.wait_for_selector(css, state="visible", timeout=deadline_step)
            if el is not None:
                return el, css
        except PWTimeout:
            continue
    return None, None


def _is_disabled(el, cfg: dict) -> bool:
    paging = cfg["paging"]
    if "disabled_attr" in paging:
        if el.get_attribute(paging["disabled_attr"]) is not None:
            return True
        if el.evaluate("e => e.disabled === true"):
            return True
    if "disabled_class" in paging:
        cls = el.get_attribute("class") or ""
        if paging["disabled_class"] in cls.split():
            return True
    return False


def _row_fingerprint(row, cells_cfg: dict) -> str:
    parts = []
    for key, sel in cells_cfg.items():
        try:
            el = row.query_selector(sel)
            parts.append((el.inner_text() if el else "").strip())
        except Exception:
            parts.append("")
    return "|".join(parts)


def _extract_row(row, cfg: dict) -> dict:
    raw: dict[str, str] = {}
    for key, sel in cfg["txn"]["cells"].items():
        el = row.query_selector(sel)
        raw[key] = el.inner_text().strip() if el else ""
    href_el = row.query_selector(cfg["receipt"]["link_selector"])
    raw["receipt"] = href_el.get_attribute("href") if href_el else ""
    return normalize_row(raw, cfg)


def _read_balance(page, cfg: dict) -> float | None:
    try:
        el = page.wait_for_selector(_resolve(cfg["balance"]["selector"]),
                                    state="attached", timeout=3000)
        text = (el.inner_text() or el.get_attribute("textContent") or "").strip()
        from .parser import parse_amount
        return parse_amount(text)
    except Exception:
        return None


def _download_receipt(context, base_url: str, href: str,
                      txn_ref: str, bank_code: str, txn_path: str) -> tuple[Path | None, dict | None]:
    """下载回单 PDF。

    headless chromium 访问 PDF 链接可能直接触发浏览器下载（无 PDF 查看器插件），
    因此统一在已登录的页面上下文里用 fetch() 取二进制，会话 cookie 自动带上，
    对 inline 展示和 attachment 下载两种响应头都成立。
    """
    if not href:
        return None, None
    url = urllib.parse.urljoin(base_url, href)
    page = context.new_page()
    try:
        # 先落到同域页面，保证会话 cookie 与同源策略都满足，再在页面内 fetch PDF
        page.goto(urllib.parse.urljoin(base_url, txn_path), wait_until="domcontentloaded",
                  timeout=15000)
        pdf_bytes = page.evaluate(
            """async (u) => {
                const resp = await fetch(u, {credentials: 'include'});
                if (!resp.ok) return null;
                const buf = await resp.arrayBuffer();
                return Array.from(new Uint8Array(buf));
            }""",
            url,
        )
        if not pdf_bytes:
            return None, None
        pdf_bytes = bytes(pdf_bytes)
    finally:
        page.close()

    safe_ref = txn_ref.replace("/", "_")
    target = config.RECEIPT_DIR / bank_code / f"{safe_ref}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf_bytes)
    try:
        fields = extract_receipt_fields(pdf_bytes)
    except Exception:
        fields = {"payer": None, "purpose": None, "raw_text": ""}
    return target, fields


def run_download(base_url: str, bank_code: str, cfg: dict, username: str, password: str,
                 date_from: str, date_to: str, token_url: str,
                 headless: bool = True, save_receipts: bool = True) -> dict:
    """执行一次完整下载，返回标准结构（不落库，落库由 tasks 层负责）。"""
    config.ensure_dirs()
    timeout_ms = cfg.get("timeout_ms", 15000)
    seen_refs: set[str] = set()
    txns: list[dict] = []
    balance: float | None = None
    receipts_saved = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            # 1. 登录（动态口令走令牌接口）
            page.goto(urllib.parse.urljoin(base_url, cfg["login"]["path"]))
            page.fill(_resolve(cfg["login"]["user_selector"]), username)
            page.fill(_resolve(cfg["login"]["pwd_selector"]), password)
            otp = fetch_otp(token_url, username)
            page.fill(_resolve(cfg["login"]["otp_selector"]), otp)
            page.click(_resolve(cfg["login"]["submit_selector"]))
            page.wait_for_url(f"**{cfg['login']['landing_path']}", timeout=timeout_ms)

            # 2. 带日期条件打开明细页
            qs = urllib.parse.urlencode({
                cfg["txn"]["date_from_param"]: date_from,
                cfg["txn"]["date_to_param"]: date_to,
            })
            page.goto(urllib.parse.urljoin(base_url, f"{cfg['txn']['path']}?{qs}"))
            # 明细页可能零行（增量窗口落在无流水日期）：行选择器等不到时，
            # 用恒在的余额元素确认页面已加载，按空结果页处理而不是判失败。
            try:
                page.wait_for_selector(_resolve(cfg["txn"]["row_selector"]), timeout=timeout_ms)
            except PWTimeout:
                page.wait_for_selector(_resolve(cfg["balance"]["selector"]),
                                       state="attached", timeout=timeout_ms)
            balance = _read_balance(page, cfg)

            # 3. 翻页抓取，内容指纹防死循环
            page_no = 1
            fingerprints: set[str] = set()
            while page_no <= cfg.get("max_pages", 50):
                rows = page.query_selector_all(_resolve(cfg["txn"]["row_selector"]))
                if not rows:
                    break
                page_fp = "##".join(_row_fingerprint(r, cfg["txn"]["cells"]) for r in rows)
                if page_fp in fingerprints:
                    raise RpaError(f"{bank_code} 第 {page_no} 页内容与之前重复，翻页失效，终止以防死循环")
                fingerprints.add(page_fp)

                for row in rows:
                    txn = _extract_row(row, cfg)
                    if not txn["txn_ref"]:
                        continue
                    if txn["txn_ref"] in seen_refs:
                        continue  # 银行侧重复行，直接跳过
                    seen_refs.add(txn["txn_ref"])
                    txns.append(txn)

                next_el, next_css = _first_visible(page, cfg["paging"]["next_candidates"], timeout_ms)
                if next_el is None or _is_disabled(next_el, cfg):
                    break
                next_el.click()
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
                page.wait_for_selector(_resolve(cfg["txn"]["row_selector"]), timeout=timeout_ms)
                page_no += 1

            # 4. 逐笔下载并解析回单 PDF
            if save_receipts:
                for txn in txns:
                    path, fields = _download_receipt(
                        context, base_url, txn.pop("receipt_href", ""),
                        txn["txn_ref"], bank_code, cfg["txn"]["path"])
                    if path is not None:
                        txn["receipt_path"] = str(path)
                        txn["receipt_payer"] = (fields or {}).get("payer")
                        txn["receipt_purpose"] = (fields or {}).get("purpose")
                        receipts_saved += 1
        finally:
            context.close()
            browser.close()

    return {
        "bank_code": bank_code,
        "date_from": date_from,
        "date_to": date_to,
        "balance": balance,
        "txns": txns,
        "rows_fetched": len(txns),
        "receipts_saved": receipts_saved,
    }
