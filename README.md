# lmsched — 投机解码推理调度核心

纯 Go 标准库实现的 LLM 推理调度原型：一个固定种子随机初始化的微型
Transformer（目标模型 + 更小的草稿模型），重点在**调度与解码算法**，
而非模型精度。所有随机数（初始化、采样）都由可复现的 PCG32 驱动。

## 目录结构

```
lm/        微型 Transformer：可复现随机初始化、分页 KV、任意注意力掩码、
           树形掩码一次前向、读视图/写槽分离的 KV 抽象
sample/    逐请求采样：温度、Top-P 核采样、重复惩罚、投机拒绝采样与修正分布
engine/    调度核心：连续批处理、分块预填充、换页、链式/树形投机解码、
           自适应 k、工作负载生成、监控快照与嵌入仪表盘
cmd/server  HTTP 服务 + 实时监控页面
cmd/bench   不同 k / 批大小 / 显存压力下的吞吐与首字延迟基准
```

## 快速开始

```bash
# 全量单元测试（含全部 8 条需求的验收测试，26 个测试用例）
go test ./... -v

# 启动监控服务（默认 :8080，浏览器打开看实时调度）
go run ./cmd/server -addr :8080 -autoload 12 -spec chain

# 跑基准：对比不同投机 k、批大小、显存受限（换页）场景
go run ./cmd/bench -requests 120
```

服务端点：`GET /` 仪表盘、`GET /state` 状态 JSON、`POST /submit` 手动提交请求。

## 八条需求与验收测试对照

| # | 需求 | 关键测试 |
|---|------|----------|
| 1 | 连续批处理，每步换入换出不等整批 | `TestContinuousBatching` |
| 2 | 树形验证，取最长接受路径 | `TestTreeChainAcceptIdentical`（树形与链式接受词完全一致） |
| 3 | 同批逐请求独立温度/TopP/重复惩罚 | `TestPerRequestSamplingParams`、`TestSamplingParamsTakeEffect` |
| 4 | 投机拒绝采样分布与目标一致，拒绝后修正重采 | `TestSpeculativeDistributionMatchesTarget`（卡方）、`TestEndToEndSamplingDistribution`、`TestResampleNorm` |
| 5 | k 按最近接受率自适应 | `TestAdaptiveK`、`TestAdjustKDirection` |
| 6 | 显存不足整块换出/换回，不重算前缀 | `TestSwapPreservesOutput`、`TestSwapKeepsBlockTable`、`TestSwapPreservesKV` |
| 7 | 长输入分块预填充，与短请求解码交替 | `TestPrefillChunkInvariance`、`TestShortRequestNotBlocked` |
| 8 | 贪心下投机与逐词解码逐词一致 | `TestGreedyEquivalence`、`TestGreedyInvariantAcrossK` |

## 关键设计决策

- **读视图 / 写槽分离（Segment.ReadKV 与 WriteSink）**：验证时目标模型读
  “分页前缀 + 已接受瞬态 KV”，但把自己的 K/V 写到独立缓冲，落盘的永远是
  目标模型的 K/V，而不是草稿模型的——这是投机解码输出与纯目标解码严格
  一致的基础。
- **位置对齐的拒绝采样**：用“上一个被接受词的目标前向输出”验证当前草稿
  词，接受后才对该词跑目标前向，保证分布与纯目标采样逐位置相同。
- **GPU 固定槽 + CPU 按“请求+逻辑块”暂存**：换出时深拷贝块数据并释放物理
  槽，换回时分到任意空闲槽并重写块表，逻辑位置与内容不变，物理槽可安全复用。
- **换页防活锁**：待换回请求优先、新请求在有换出债务时不得侵占块、单 tick
  抢占预算受限，并提供 `RunWithGuard` 无进展保护。

## 基准指标解读

`tok/tick` 是调度吞吐（与墙钟算力解耦），`TTFT` 为首字延迟（tick）。
典型现象：批大小翻倍吞吐近似线性提升；显存受限时换页使 TTFT 上升、
吞吐下降，但所有请求输出与显存充足时逐词一致。
