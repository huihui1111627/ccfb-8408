"""测试夹具：临时数据目录 + mock 令牌/网银服务 + 已播种数据库。

所有测试与真实文件系统/端口隔离，使用独立高位端口，避免与本地演示环境冲突。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 必须在导入 bankrpa 之前定好环境变量
TEST_DATA = Path("/tmp/bankrpa_pytest_data")
os.environ["BANKRPA_DATA_DIR"] = str(TEST_DATA)
os.environ["BANKRPA_TOKEN_PORT"] = "19300"
os.environ["BANKRPA_BANK_PORTS"] = "19401,19402,19403"
os.environ["BANKRPA_TOKEN_URL"] = "http://127.0.0.1:19300/token"

from bankrpa import config, db  # noqa: E402
from bankrpa.mock_bank.bank_app import create_app as bank_app  # noqa: E402
from bankrpa.mock_bank.data import BANKS  # noqa: E402
from bankrpa.mock_bank.token_service import create_app as token_app  # noqa: E402
from bankrpa.seed import seed  # noqa: E402

PORTS = {"ecd": 19401, "scb": 19402, "wcb": 19403}


@pytest.fixture(scope="session")
def mock_servers():
    servers = []
    token_server = uvicorn.Server(uvicorn.Config(
        token_app(), host="127.0.0.1", port=19300, log_level="warning"))
    threading.Thread(target=token_server.run, daemon=True).start()
    for code, port in PORTS.items():
        s = uvicorn.Server(uvicorn.Config(
            bank_app(code), host="127.0.0.1", port=port, log_level="warning"))
        servers.append(s)
        threading.Thread(target=s.run, daemon=True).start()
    for s in servers:
        while not s.started:
            time.sleep(0.05)
    # 等待令牌服务
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen("http://127.0.0.1:19300/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield PORTS


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture()
def conn(db_path):
    """每个测试一个全新的临时数据库（种子照灌）。"""
    db.init_db(db_path)
    seed(db_path)
    c = db.connect(db_path)
    yield c
    c.close()


@pytest.fixture()
def account_id(conn):
    def _get(bank_code: str) -> int:
        return db.query_one(conn, "SELECT id FROM bank_account WHERE bank_code=?",
                            (bank_code,))["id"]
    return _get
