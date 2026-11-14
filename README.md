# 网银对账 RPA

集团财务共享中心的网银流水下载与应收自动对账系统：Python + Playwright 驱动 mock 网银，
FastAPI + Jinja2 提供网页控制台，SQLite 落库，全流程配置化、可审计、强幂等。

## 功能全景

| 需求 | 实现位置 |
| --- | --- |
| 各家登录/下载流程可配 | `bankrpa/bank_configs.yaml`（引擎零硬编码选择器） |
| 动态口令取令牌接口 | `mock_bank/token_service.py`（TOTP，30 秒滚动）+ `rpa/engine.py::fetch_otp` |
| 增量下载、跨天补漏 | `services/tasks.py`（窗口推算 + `INSERT OR IGNORE` 安全重跑） |
| 各家流水统一解析 | `rpa/parser.py`（split/signed 两种金额模式归一） |
| 回单 PDF 抽付款方/用途 | `rpa/receipt.py`（pdfplumber，兼容多种回单表述） |
| 一对一/一对多/多对一/部分付款 | `services/reconcile.py::auto_match` |
| 配不上进差异、人工核销留痕 | `diff_item` 表 + `audit_log` 审计表 |
| 每日余额调节表 | `services/report.py` |
| 同一笔流水不重复入库/核销 | 数据库唯一约束 + 单事务写锁（见下“两条红线”） |

## 两条硬性幂等红线

1. **流水不重复入库**：`bank_tx UNIQUE(bank_code, txn_ref)` + `INSERT OR IGNORE`，
   任务重跑、跨天补漏、并发插入都只保留一笔。
2. **核销不重复发生**：`match_entry` 条件唯一索引
   `(txn_id, receivable_id) WHERE status='active'`，核销统一走
   `BEGIN IMMEDIATE` 单事务并校验双方剩余余额；8 线程并发核销同一对，只有 1 笔成功
   （测试 `test_concurrent_writeoff_only_one_wins`）。

## 启动

```bash
# 1. 令牌服务(9100) + 三家 mock 网银(9201-9203)
python -m bankrpa.mock_bank.serve

# 2. 另开一个终端：网页控制台(8000)，首次启动自动建表灌种子
python -m bankrpa.web.run_web
```

打开 http://127.0.0.1:8000 ：银行账户页点“增量下载” → 自动对账页执行自动配对 →
差异处理页人工核销 → 余额调节表出表。所有端口与数据目录可用环境变量覆盖
（`BANKRPA_DATA_DIR` / `BANKRPA_WEB_PORT` / `BANKRPA_TOKEN_PORT` / `BANKRPA_BANK_PORTS`）。

## 测试

```bash
python -m pytest          # 39 个，含真实 chromium 下载与并发幂等
```

## 目录结构

```
bankrpa/
├── bank_configs.yaml        # 三家银行全流程配置（选择器/翻页/金额模式）
├── config.py db.py seed.py  # 配置、SQLite 约束、种子数据
├── mock_bank/               # 令牌服务 + 三家 DOM 各异的 mock 网银 + PDF 回单
├── rpa/                     # 配置加载、统一解析、PDF 抽取、Playwright 引擎
├── services/                # 下载任务、对账核销、差异、余额调节表
└── web/                     # FastAPI 路由 + Jinja2 模板（8 个页面）
tests/                       # 解析/PDF、两条红线、对账场景、Web、令牌
```
