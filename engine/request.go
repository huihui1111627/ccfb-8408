package engine

import (
	"fmt"

	"lmsched/lm"
	"lmsched/sample"
)

type State int

const (
	StateWaiting  State = iota // 排队等显存块（含被换出的请求）
	StatePrefill               // 已上卡，预填充未完成
	StateDecoding              // 解码中
	StateFinished
)

func (s State) String() string {
	switch s {
	case StateWaiting:
		return "waiting"
	case StatePrefill:
		return "prefill"
	case StateDecoding:
		return "decoding"
	case StateFinished:
		return "finished"
	default:
		return "unknown"
	}
}

// Request 引擎里一个推理请求的全部运行时状态。
type Request struct {
	ID           uint64
	Prompt       []int
	MaxNewTokens int
	Params       sample.Params
	Priority     int
	Seed         uint64

	state State

	// 分页 KV（在 GPU/CPU 池之间整体迁移）与逻辑提交长度
	kv         *lm.PagedKV
	prefillPos int         // 预填充已处理到 prompt 的下标
	draftKV    *lm.SliceKV // 草稿模型瞬态 KV
	targetKV   *lm.SliceKV // 目标模型验证时的瞬态 KV（接受后落盘的是它）

	// 目标模型在最后一个已提交位置上的原始 logits（预测下一位置）
	lastTargetLogits []float64

	// 两个独立随机流：草稿采样 / 目标采样与接受判定。
	// 换出换回不改变 RNG 状态，因此输出与是否发生过换页完全无关。
	draftRNG  *lm.PCG32
	sampleRNG *lm.PCG32

	Output       []int
	finishReason string

	// 时间线（引擎 tick 口径）
	arrivedTick    int64
	admittedTick   int64
	firstTokenTick int64
	finishTick     int64

	// 投机解码统计
	kMin, kMax          int
	currentK            int
	acceptEMA           float64
	emittedSinceKAdjust int
	totalDrafts         int64
	totalAccepted       int64
	treeLongest         int // 最近一次树验证的最长接受路径（展示用）
	everSwapped         bool
	swapBackWanted      bool // 换出后想优先换回

	// 出字速度：记录最近若干次出字发生的 tick
	emitTicks []int64

	// 工作量统计
	targetForwards int64
	draftForwards  int64
	targetUnits    int64
	draftUnits     int64
}

func (r *Request) State() State { return r.state }

// SeqLen 已提交 token 总数（提示前缀 + 已生成）。
func (r *Request) SeqLen() int {
	if r.kv == nil {
		return 0
	}
	return r.kv.Length
}

// History 重复惩罚使用的完整历史。
func (r *Request) History() []int {
	h := make([]int, 0, len(r.Prompt)+len(r.Output))
	h = append(h, r.Prompt...)
	h = append(h, r.Output...)
	return h
}

func (r *Request) acceptanceRate() float64 {
	if r.totalDrafts == 0 {
		return 0
	}
	return float64(r.totalAccepted) / float64(r.totalDrafts)
}

// TokenRate 最近窗口内每个 tick 的出字数。
func (r *Request) TokenRate(nowTick int64, window int64) float64 {
	if len(r.emitTicks) == 0 {
		return 0
	}
	cutoff := nowTick - window
	var kept []int64
	for _, t := range r.emitTicks {
		if t >= cutoff {
			kept = append(kept, t)
		}
	}
	if len(kept) == 0 {
		return 0
	}
	span := nowTick - kept[0] + 1
	return float64(len(kept)) / float64(span)
}

func (r *Request) recordEmit(tick int64, n int) {
	for i := 0; i < n; i++ {
		r.emitTicks = append(r.emitTicks, tick)
		if len(r.emitTicks) > 32 {
			r.emitTicks = r.emitTicks[1:]
		}
	}
}

func (r *Request) label() string {
	return fmt.Sprintf("req#%d", r.ID)
}

// 以下访问器供基准测试与外部统计读取引擎内部指标。
func (r *Request) AdmittedTick() int64   { return r.admittedTick }
func (r *Request) FirstTokenTick() int64 { return r.firstTokenTick }
func (r *Request) TargetForwards() int64 { return r.targetForwards }
func (r *Request) DraftForwards() int64  { return r.draftForwards }
func (r *Request) EverSwapped() bool     { return r.everSwapped }

// AcceptanceRate 导出的接受率访问器（基准测试用）。
func (r *Request) AcceptanceRate() float64 { return r.acceptanceRate() }
