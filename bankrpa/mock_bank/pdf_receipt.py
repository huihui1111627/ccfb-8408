"""回单 PDF 生成：使用 reportlab 内置 CID 中文字体 STSong-Light，无需外部字体文件。

PDF 内固定包含付款方与用途两个字段，供 receipt.py 抽取验证。
"""
from __future__ import annotations

import io

from reportlab.lib.pagesizes import A5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

_FONT = "STSong-Light"
_registered = False


def _ensure_font() -> None:
    global _registered
    if not _registered:
        pdfmetrics.registerFont(UnicodeCIDFont(_FONT))
        _registered = True


def build_receipt_pdf(bank_name: str, txn: dict) -> bytes:
    _ensure_font()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A5)
    width, height = A5
    y = height - 60

    c.setFont(_FONT, 16)
    c.drawCentredString(width / 2, y, f"{bank_name}电子回单")
    y -= 34
    c.setFont(_FONT, 11)
    lines = [
        f"流水号：{txn['ref']}",
        f"交易时间：{txn['time']}",
        f"交易金额：{txn['amount']:.2f} 元",
        f"收款方/付款方：{txn['counterparty']}",
        f"付款方：{txn['payer']}",
        f"用途：{txn['purpose']}",
        f"摘要：{txn['summary']}",
    ]
    for line in lines:
        c.drawString(50, y, line)
        y -= 24
    c.setFont(_FONT, 9)
    c.drawString(50, 40, "本回单由 mock 网银生成，仅用于 RPA 联调测试。")
    c.showPage()
    c.save()
    return buf.getvalue()
