"""加载 bank_configs.yaml，做基本结构校验。"""
from __future__ import annotations

from functools import lru_cache

import yaml

from ..config import BANK_CONFIGS_PATH

_REQUIRED = ("login", "txn", "paging")


@lru_cache(maxsize=8)
def load_configs(path: str | None = None) -> dict[str, dict]:
    target = path or str(BANK_CONFIGS_PATH)
    with open(target, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    banks = raw["banks"]
    for code, cfg in banks.items():
        for key in _REQUIRED:
            if key not in cfg and not (key == "amount" and "amount" in cfg.get("txn", {})):
                raise ValueError(f"银行配置 {code} 缺少必填段: {key}")
        if "row_selector" not in cfg["txn"]:
            raise ValueError(f"银行配置 {code} 缺少 txn.row_selector")
        if cfg["txn"]["amount"]["mode"] not in ("split", "signed"):
            raise ValueError(f"银行配置 {code} 的 amount.mode 非法")
    return banks


def get_bank_config(bank_code: str, path: str | None = None) -> dict:
    banks = load_configs(path)
    if bank_code not in banks:
        raise KeyError(f"未找到银行 {bank_code} 的流程配置")
    return banks[bank_code]
