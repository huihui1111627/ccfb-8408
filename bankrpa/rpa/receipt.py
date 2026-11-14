"""回单 PDF 抽取：从回单文本里提取付款方与用途。

mock 回单固定带“付款方：xxx”“用途：xxx”两行；解析器同时兼容
“付款人/付款账户名称”“款项用途/摘要用途”等常见真实回单表述。
"""
from __future__ import annotations

import re

import pdfplumber

_PAYER_PATTERNS = [
    re.compile(r"付\s*款\s*方[：:]\s*(.+)"),
    re.compile(r"付\s*款\s*人[：:]\s*(.+)"),
    re.compile(r"付款账户名称[：:]\s*(.+)"),
]
_PURPOSE_PATTERNS = [
    re.compile(r"用\s*途[：:]\s*(.+)"),
    re.compile(r"款项用途[：:]\s*(.+)"),
    re.compile(r"摘要用途[：:]\s*(.+)"),
]


def _match_any(lines: list[str], patterns: list[re.Pattern]) -> str | None:
    for line in lines:
        for p in patterns:
            m = p.search(line.strip())
            if m:
                return m.group(1).strip()
    return None


def extract_receipt_fields(pdf_bytes: bytes) -> dict:
    """返回 {'payer': ..., 'purpose': ..., 'raw_text': ...}，抽不到给 None 不报错。"""
    text_parts: list[str] = []
    with pdfplumber.open(_as_filelike(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    raw_text = "\n".join(text_parts)
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    return {
        "payer": _match_any(lines, _PAYER_PATTERNS),
        "purpose": _match_any(lines, _PURPOSE_PATTERNS),
        "raw_text": raw_text,
    }


def _as_filelike(pdf_bytes: bytes):
    import io
    return io.BytesIO(pdf_bytes)
