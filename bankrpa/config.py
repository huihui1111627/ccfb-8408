"""全局路径与运行参数（均可通过环境变量覆盖，测试用临时目录隔离）。"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("BANKRPA_DATA_DIR", PACKAGE_DIR.parent / "bankrpa_data"))
RECEIPT_DIR = DATA_DIR / "receipts"
DB_PATH = DATA_DIR / "bankrpa.db"
BANK_CONFIGS_PATH = PACKAGE_DIR / "bank_configs.yaml"

TOKEN_URL = os.environ.get("BANKRPA_TOKEN_URL", "http://127.0.0.1:9100/token")
TOKEN_PORT = int(os.environ.get("BANKRPA_TOKEN_PORT", "9100"))
WEB_PORT = int(os.environ.get("BANKRPA_WEB_PORT", "8000"))
# 三家 mock 网银的默认端口，逗号分隔
BANK_PORTS = [int(p) for p in os.environ.get("BANKRPA_BANK_PORTS", "9201,9202,9203").split(",")]

# 金额比对容差（分以下的浮点误差）
AMOUNT_EPS = 0.005


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
