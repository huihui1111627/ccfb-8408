"""控制台启动入口：python -m bankrpa.web.run_web"""
from __future__ import annotations

import uvicorn

from .. import config
from ..seed import seed
from .app import app


def main() -> None:
    config.ensure_dirs()
    seed()  # 启动自动建表灌种子，重复执行幂等
    uvicorn.run(app, host="127.0.0.1", port=config.WEB_PORT, log_level="info")


if __name__ == "__main__":
    main()
