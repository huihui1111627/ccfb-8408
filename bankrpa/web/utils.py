"""Web 工具：修复 Starlette 表单把中文按 latin-1 解码的问题。

浏览器以 UTF-8 发送 application/x-www-form-urlencoded，Starlette 历史上默认
按 latin-1 解码 query/body 字段，中文会变 mojibake。统一在取表单后过一遍
fix_form_utf8 修正，所有含中文的 POST 都必须使用。
"""
from __future__ import annotations


def fix_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value  # 本来就是正确的 utf-8，原样返回


def fix_form_utf8(form) -> dict[str, str]:
    return {k: fix_text(v) for k, v in form.items()}
