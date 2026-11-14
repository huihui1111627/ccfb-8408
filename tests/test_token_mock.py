"""mock 令牌服务与网银登录安全的测试。"""
from __future__ import annotations

import urllib.request

import pytest

from bankrpa.mock_bank.token_service import current_code, verify_code
from bankrpa.rpa.engine import fetch_otp


def test_token_service_roundtrip(mock_servers):
    url = "http://127.0.0.1:19300/token"
    otp = fetch_otp(url, "ecd_corp")
    assert len(otp) == 6 and otp.isdigit()
    assert verify_code("ecd_corp", otp) is True
    assert verify_code("ecd_corp", "000000") is False


def test_token_unknown_account(mock_servers):
    with pytest.raises(Exception):
        fetch_otp("http://127.0.0.1:19300/token", "nobody")


def test_otp_is_time_based():
    c1 = current_code("scb_corp", at=1_000_000)
    c2 = current_code("scb_corp", at=1_000_030)  # 下一个 30 秒窗口
    assert c1 != c2
    assert current_code("scb_corp", at=1_000_000) == c1  # 同窗口稳定


def test_bank_login_rejects_bad_otp(mock_servers):
    # 用 urllib 直接打登录表单：错误口令必须被拒
    import urllib.parse
    data = urllib.parse.urlencode({
        "username": "wcb_corp", "password": "Wcb@123456", "otp": "000000",
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:19403/login", data=data)
    try:
        urllib.request.urlopen(req)
        assert False, "错误口令不应登录成功"
    except urllib.error.HTTPError as e:
        assert e.code == 401


def test_bank_transactions_requires_session(mock_servers):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as ei:
        opener.open("http://127.0.0.1:19401/transactions")
    assert ei.value.code == 302
