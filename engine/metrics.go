package engine

import "lmsched/lm"

// ReqSnapshot 页面与基准测试使用的请求级快照。
type ReqSnapshot struct {
	ID           uint64  `json:"id"`
	State        string  `json:"state"`
	Priority     int     `json:"priority"`
	PromptLen    int     `json:"prompt_len"`
	Generated    int     `json:"generated"`
	MaxNewTokens int     `json:"max_new_tokens"`
	K            int     `json:"k"`
	AcceptRate   float64 `json:"accept_rate"`
	AcceptEMA    float64 `json:"accept_ema"`
	TokenRate    float64 `json:"token_rate"`
	TreeLongest  int     `json:"tree_longest"`
	EverSwapped  bool    `json:"ever_swapped"`
	SwapBackWant bool    `json:"swap_back_wanted"`
	TTFT         int64   `json:"ttft_ticks"`
	Greedy       bool    `json:"greedy"`
	Temperature  float64 `json:"temperature"`
	TopP         float64 `json:"top_p"`
	RepPenalty   float64 `json:"rep_penalty"`
	Blocks       int     `json:"blocks"`
	FinishReason string  `json:"finish_reason,omitempty"`
	Preview      []int   `json:"preview"`
}

// BlockCell 显存块占用图里的一格。
type BlockCell struct {
	PhysicalID int    `json:"physical_id"`
	Pool       string `json:"pool"`       // gpu / cpu
	RequestID  uint64 `json:"request_id"` // 0 表示空闲
}

// Snapshot 引擎全量状态快照。
type Snapshot struct {
	Tick int64 `json:"tick"`

	SpecMode  string `json:"spec_mode"`
	BlockSize int    `json:"block_size"`
	GPUBlocks int    `json:"gpu_blocks"`
	CPUBlocks int    `json:"cpu_blocks"`
	GPUFree   int    `json:"gpu_free"`
	CPUFree   int    `json:"cpu_free"`
	SwapBytes int64  `json:"swap_bytes"`
	SwapsOut  int    `json:"swaps_out"`
	SwapsIn   int    `json:"swaps_in"`

	Decoding []ReqSnapshot `json:"decoding"`
	Prefill  []ReqSnapshot `json:"prefill"`
	Waiting  []ReqSnapshot `json:"waiting"`
	Finished []ReqSnapshot `json:"finished"`

	Blocks []BlockCell `json:"blocks"`

	TotalTargetUnits int64 `json:"total_target_units"`
	TotalDraftUnits  int64 `json:"total_draft_units"`
	TotalTargetFwd   int64 `json:"total_target_forwards"`
	TotalDraftFwd    int64 `json:"total_draft_forwards"`
}

func (e *Engine) reqSnapshot(r *Request) ReqSnapshot {
	s := ReqSnapshot{
		ID:           r.ID,
		State:        r.state.String(),
		Priority:     r.Priority,
		PromptLen:    len(r.Prompt),
		Generated:    len(r.Output),
		MaxNewTokens: r.MaxNewTokens,
		K:            r.currentK,
		AcceptRate:   r.acceptanceRate(),
		AcceptEMA:    r.acceptEMA,
		TokenRate:    r.TokenRate(e.tick, e.cfg.StatsWindowTicks),
		TreeLongest:  r.treeLongest,
		EverSwapped:  r.everSwapped,
		SwapBackWant: r.swapBackWanted,
		Greedy:       r.Params.Greedy,
		Temperature:  r.Params.Temperature,
		TopP:         r.Params.TopP,
		RepPenalty:   r.Params.RepetitionPenalty,
		FinishReason: r.finishReason,
	}
	if r.kv != nil {
		s.Blocks = len(r.kv.Table)
	}
	if r.firstTokenTick > 0 {
		s.TTFT = r.firstTokenTick - r.admittedTick
	}
	preview := 8
	if len(r.Output) < preview {
		preview = len(r.Output)
	}
	s.Preview = append([]int(nil), r.Output[:preview]...)
	return s
}

// Snapshot 取当前状态的只读快照（与 Step 同 goroutine 调用）。
func (e *Engine) Snapshot() Snapshot {
	snap := Snapshot{
		Tick:             e.tick,
		SpecMode:         e.cfg.Spec.String(),
		BlockSize:        e.cfg.BlockSize,
		GPUBlocks:        e.bm.GPUTotal(),
		CPUBlocks:        e.bm.CPUTotal(),
		GPUFree:          e.bm.GPUFree(),
		CPUFree:          e.bm.CPUFree(),
		SwapBytes:        e.bm.SwapBytes,
		SwapsOut:         e.bm.SwapsOut,
		SwapsIn:          e.bm.SwapsIn,
		TotalTargetUnits: e.totalTargetUnits,
		TotalDraftUnits:  e.totalDraftUnits,
		TotalTargetFwd:   e.totalTargetFwd,
		TotalDraftFwd:    e.totalDraftFwd,
	}
	for _, r := range e.active {
		rs := e.reqSnapshot(r)
		switch r.state {
		case StatePrefill:
			snap.Prefill = append(snap.Prefill, rs)
		case StateDecoding:
			snap.Decoding = append(snap.Decoding, rs)
		}
	}
	for _, r := range e.waiting.ordered() {
		snap.Waiting = append(snap.Waiting, e.reqSnapshot(r))
	}
	// finished 只展示最近 50 条，避免页面无限增长
	start := 0
	if len(e.finished) > 50 {
		start = len(e.finished) - 50
	}
	for _, r := range e.finished[start:] {
		snap.Finished = append(snap.Finished, e.reqSnapshot(r))
	}

	// 显存块占用图：GPU 在前 CPU 在后，标注占用请求
	owner := make(map[int]uint64)
	for id, table := range e.bm.Allocs() {
		for _, phys := range table {
			owner[phys] = id
		}
	}
	for _, id := range e.bm.GPUBlockIDs() {
		snap.Blocks = append(snap.Blocks, BlockCell{
			PhysicalID: id, Pool: "gpu", RequestID: owner[id],
		})
	}
	for _, id := range e.bm.CPUBlockIDs() {
		snap.Blocks = append(snap.Blocks, BlockCell{
			PhysicalID: id, Pool: "cpu", RequestID: e.bm.CPUOwners()[id],
		})
	}
	return snap
}

// Run 推进直到所有请求完成，返回总 tick 数。
// maxTicks>0 时作为硬上限，防止极端资源配置下换页活锁导致无限循环。
func (e *Engine) Run(maxTicks int) int {
	for !e.Quiescent() {
		if maxTicks > 0 && e.tick >= int64(maxTicks) {
			break
		}
		e.Step()
	}
	return int(e.tick)
}

// RunWithGuard 推进直到完成；若连续 stallTicks 个 tick 已完成请求数都没有
// 增长（疑似换页活锁），则提前返回当前 tick 与是否收敛。
func (e *Engine) RunWithGuard(stallTicks int) (ticks int, converged bool) {
	lastFinished := -1
	stall := 0
	for !e.Quiescent() {
		e.Step()
		nf := len(e.finished)
		if nf == lastFinished {
			stall++
			if stall >= stallTicks {
				return int(e.tick), false
			}
		} else {
			stall = 0
			lastFinished = nf
		}
	}
	return int(e.tick), true
}

// BlockManager 暴露给基准/测试使用。
func (e *Engine) BlockManager() *lm.BlockManager { return e.bm }

// Tick 当前 tick。
func (e *Engine) Tick() int64 { return e.tick }

// Finished 已完成请求（测试断言用）。
func (e *Engine) Finished() []*Request { return e.finished }
