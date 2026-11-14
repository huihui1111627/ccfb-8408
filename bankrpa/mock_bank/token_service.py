"""mock 动态令牌服务：每个登录用户一个 TOTP 密钥，30 秒滚动 6 位口令。

RPA 引擎只允许通过本服务的 HTTP 接口取口令（GET /token），
mock 网银登录时用同一算法本地校验，保证两边始终同步。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .data import BANKS

# 每家银行固定一个 base32 密钥（mock 用途，不涉及真实凭据）
_SECRETS = {
    code: base64.b32encode(f"bankrpa-secret-{code}".encode()).decode()
    for code in BANKS
}


def _hotp(secret_b32: str, counter: int) -> str:
    key = base64.b32decode(secret_b32)
    msg = counter.to_bytes(8, "big")
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = int.from_bytes(digest[offset:offset + 4], "big") & 0x7FFFFFFF
    return f"{binary % 1_000_000:06d}"


def current_code(login_user: str, at: float | None = None) -> str | None:
    bank_code = next((c for c, b in BANKS.items() if b["login_user"] == login_user), None)
    if bank_code is None:
        return None
    counter = int((at or time.time()) // 30)
    return _hotp(_SECRETS[bank_code], counter)


def verify_code(login_user: str, code: str, at: float | None = None) -> bool:
    expected = current_code(login_user, at)
    return expected is not None and hmac.compare_digest(expected, str(code))


def create_app() -> FastAPI:
    app = FastAPI(title="mock 动态令牌服务")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/token")
    def token(account: str):
        code = current_code(account)
        if code is None:
            return JSONResponse({"error": "unknown account"}, status_code=404)
        return {"account": account, "otp": code, "ttl_seconds": 30}

    @app.get("/verify")
    def verify(account: str, code: str):
        ok = verify_code(account, code)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 401)

    return app


app = create_app()
