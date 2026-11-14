"""一键启动令牌服务 + 三家 mock 网银（端口 9100 / 9201-9203）。"""
from __future__ import annotations

import uvicorn

from .. import config
from .bank_app import create_app as bank_app
from .data import BANKS
from .token_service import create_app as token_app


def main() -> None:
    config.ensure_dirs()
    token_server = uvicorn.Server(uvicorn.Config(token_app(), host="127.0.0.1",
                                                 port=config.TOKEN_PORT, log_level="warning"))
    token_server.config.load()
    token_server.lifespan = token_server.config.lifespan_class(token_server.config)
    import threading
    threading.Thread(target=token_server.run, daemon=True).start()

    servers = []
    for bank_code, port in zip(BANKS, config.BANK_PORTS):
        server = uvicorn.Server(uvicorn.Config(bank_app(bank_code), host="127.0.0.1",
                                               port=port, log_level="warning"))
        servers.append(server)
        threading.Thread(target=server.run, daemon=True).start()

    print(f"mock 令牌服务: http://127.0.0.1:{config.TOKEN_PORT}/token?account=<user>")
    for bank_code, port in zip(BANKS, config.BANK_PORTS):
        print(f"mock 网银 {BANKS[bank_code]['bank_name']}({bank_code}): http://127.0.0.1:{port}")
    print("按 Ctrl+C 退出。")
    try:
        for server in servers:
            while not server.started:
                threading.Event().wait(0.1)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
