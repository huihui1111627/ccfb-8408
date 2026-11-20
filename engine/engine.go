package engine

import (
	"lmsched/lm"
	"lmsched/sample"
)

// SpecMode 投机解码形态。
type SpecMode int

const (
	SpecNone  SpecMode = iota // 逐词解码
	SpecChain                 // 逐条链式验证
	SpecTree                  // 树形掩码一次前向验证
)

func (m SpecMode) String() string {
	switch m {
	case SpecChain:
		return "chain"
	case SpecTree:
		return "tree"
	default:
		return "none"
	}
}

// Config 引擎配置。
type Config struct {
	PrefillChunkSize int
	Spec             SpecMode
	KMin             int
	KMax             int
	TreeBranches     int
	BlockSize        int
	GPUBlocks        int
	CPUBlocks        int
	EmaAlpha         float64
	KAdjustWindow    int
	StatsWindowTicks int64
	MaxBatch         int
}

// Engine 调度核心。Step 是单线程推进的，页面快照也在同一 goroutine 取，
// 避免任何数据竞争；测试可以精确控制推进多少个 tick。
type Engine struct {
	target *lm.Model
	draft  *lm.Model
	bm     *lm.BlockManager
	cfg    Config

	nextID     uint64
	tick       int64
	active     map[uint64]*Request
	finished   []*Request
	waiting    *waitQueue
	roundRobin uint64 // 预填充预算轮转起点

	// 累计指标
	totalTargetUnits int64
	totalDraftUnits  int64
	totalTargetFwd   int64
	totalDraftFwd    int64
}

func NewEngine(target, draft *lm.Model, cfg Config) *Engine {
	if cfg.KMin < 1 {
		cfg.KMin = 1
	}
	if cfg.KMax < cfg.KMin {
		cfg.KMax = cfg.KMin
	}
	if cfg.TreeBranches < 1 {
		cfg.TreeBranches = 1
	}
	if cfg.EmaAlpha == 0 {
		cfg.EmaAlpha = 0.4
	}
	if cfg.KAdjustWindow == 0 {
		cfg.KAdjustWindow = 8
	}
	if cfg.StatsWindowTicks == 0 {
		cfg.StatsWindowTicks = 16
	}
	return &Engine{
		target:  target,
		draft:   draft,
		bm:      lm.NewBlockManager(target.Cfg, cfg.BlockSize, cfg.GPUBlocks, cfg.CPUBlocks),
		cfg:     cfg,
		active:  make(map[uint64]*Request),
		waiting: newWaitQueue(),
	}
}

// Submit 新请求入队（到达时间记为当前 tick）。
func (e *Engine) Submit(prompt []int, maxNew int, prio int, p sample.Params, seed uint64) *Request {
	e.nextID++
	r := &Request{
		ID:           e.nextID,
		Prompt:       append([]int(nil), prompt...),
		MaxNewTokens: maxNew,
		Params:       p,
		Priority:     prio,
		Seed:         seed,
		state:        StateWaiting,
		arrivedTick:  e.tick,
		kMin:         e.cfg.KMin,
		kMax:         e.cfg.KMax,
		currentK:     e.cfg.KMin,
	}
	e.waiting.push(r)
	return r
}

// Quiescent 所有请求都完成且队列为空。
func (e *Engine) Quiescent() bool {
	return e.waiting.len() == 0 && len(e.active) == 0
}

// Step 推进一个调度 tick：换页准入 -> 分块预填充/解码交替 -> 收尾换出完成请求。
func (e *Engine) Step() {
	e.tick++
	e.admit()

	prefilling, decoding := e.partitionActive()
	if len(prefilling) > 0 {
		e.runPrefill(prefilling)
	}
	// 预填充只占用本 tick 的一部分预算，解码请求每个 tick 仍然前进一步，
	// 短请求的首字不会被长提示拖住。
	for _, r := range decoding {
		if r.state != StateDecoding {
			continue // 同 tick 内可能刚完成预填充
		}
		e.runDecode(r)
	}
	e.reapFinished()
}

func (e *Engine) partitionActive() (prefilling, decoding []*Request) {
	for _, r := range e.active {
		switch r.state {
		case StatePrefill:
			prefilling = append(prefilling, r)
		case StateDecoding:
			decoding = append(decoding, r)
		}
	}
	return
}

// reapFinished 回收已完成请求的显存块。
func (e *Engine) reapFinished() {
	for id, r := range e.active {
		if r.state != StateFinished {
			continue
		}
		r.finishTick = e.tick
		e.bm.Free(id)
		delete(e.active, id)
		e.finished = append(e.finished, r)
	}
}

func (e *Engine) blockSize() int { return e.cfg.BlockSize }

// growIfNeeded 确保请求能容纳 seqLen 个 token 的块；缺块时触发换出抢占。
func (e *Engine) growIfNeeded(r *Request, seqLen int) bool {
	need := (seqLen + e.blockSize() - 1) / e.blockSize()
	for len(r.kv.Table) < need {
		if _, ok := e.bm.Grow(r.ID); ok {
			continue
		}
		if !e.preempt(r) {
			return false
		}
	}
	return true
}

// preempt 按优先级换出一个受害者，为 r 腾出显存块。
// 受害者选择：优先级最低；同级选最晚到达；永不选 r 自己。
func (e *Engine) preempt(r *Request) bool {
	var victim *Request
	for _, c := range e.active {
		if c.ID == r.ID || c.state == StateFinished {
			continue
		}
		if victim == nil {
			victim = c
			continue
		}
		if c.Priority < victim.Priority ||
			(c.Priority == victim.Priority && c.arrivedTick > victim.arrivedTick) ||
			(c.Priority == victim.Priority && c.arrivedTick == victim.arrivedTick && c.ID > victim.ID) {
			victim = c
		}
	}
	if victim == nil {
		return false
	}
	moved := e.bm.SwapOut(victim.ID)
	if moved == 0 {
		return false
	}
	victim.everSwapped = true
	victim.swapBackWanted = true
	victim.state = StateWaiting
	delete(e.active, victim.ID)
	e.waiting.push(victim)
	return true
}

// admit 把等待队列里能装下的请求换入/准入，受 MaxBatch 约束。
func (e *Engine) admit() {
	// 每 tick 主动抢占上限：限制换页频率，给待换回请求留出积累块的窗口，
	// 避免“换出→块被抢→再换出”的持续颠簸。
	swapBudget := 1

	for e.waiting.len() > 0 {
		if e.cfg.MaxBatch > 0 && len(e.active) >= e.cfg.MaxBatch {
			return
		}
		r := e.waiting.peek()

		if r.swapBackWanted {
			need := len(r.kv.Table)
			if need > e.bm.GPUTotal() {
				return // 结构性装不下，等其它请求结束释放
			}
			if e.bm.GPUFree() >= need {
				if e.swapInAndActivate(r) {
					continue
				}
				return
			}
			// 队首换回请求暂时装不下：优先用现有空闲块满足队列里能装下的换出请求。
			if alt := e.pickSwapBackFitting(e.bm.GPUFree()); alt != nil {
				if e.swapInAndActivate(alt) {
					continue
				}
				return
			}
			// 都装不下：在预算内抢占一次，然后本 tick 收手，下 tick 再评估。
			if swapBudget <= 0 {
				return
			}
			if !e.preempt(r) {
				return
			}
			swapBudget--
			continue
		}

		// 有待换回请求排队时，新请求不得侵占块。
		if e.hasSwapBackWaiting() {
			return
		}
		if e.bm.GPUFree() == 0 {
			if swapBudget <= 0 || !e.preempt(r) {
				return
			}
			swapBudget--
		}
		popped := e.waiting.pop()
		// 防御：堆可能在抢占中重排，若弹到的其实是待换回请求，
		// 绝不能当新请求初始化（会覆盖其 KV 句柄），重新入队并停止本 tick。
		if popped.swapBackWanted {
			e.waiting.push(popped)
			return
		}
		e.initRequest(popped)
		e.active[popped.ID] = popped
	}
}

// pickSwapBackFitting 在等待队列中找一个换出请求，其块需求不超过 freeBlocks。
// 优先选需求最小的，尽快让请求恢复解码。
func (e *Engine) pickSwapBackFitting(freeBlocks int) *Request {
	var best *Request
	for _, r := range e.waiting.ordered() {
		if !r.swapBackWanted {
			continue
		}
		if len(r.kv.Table) > freeBlocks {
			continue
		}
		if best == nil || len(r.kv.Table) < len(best.kv.Table) {
			best = r
		}
	}
	return best
}

// swapInAndActivate 把换出请求换回 GPU 并重新激活，同时从等待队列移除。
// 仅当全部块都成功换回才激活；否则保留在等待队列，下 tick 重试，避免丢失。
func (e *Engine) swapInAndActivate(r *Request) bool {
	need := len(r.kv.Table)
	if e.bm.GPUFree() < need {
		return false
	}
	moved := e.bm.SwapIn(r.ID)
	if moved != need {
		return false // 部分换回，保留等待状态
	}
	e.waiting.remove(r)
	r.swapBackWanted = false
	if r.prefillPos >= len(r.Prompt) {
		r.state = StateDecoding
	} else {
		r.state = StatePrefill
	}
	e.active[r.ID] = r
	return true
}

// hasSwapBackWaiting 队列里是否存在待换回请求。
func (e *Engine) hasSwapBackWaiting() bool {
	for _, r := range e.waiting.ordered() {
		if r.swapBackWanted {
			return true
		}
	}
	return false
}

func (e *Engine) initRequest(r *Request) {
	r.admittedTick = e.tick
	r.kv = e.bm.Acquire(r.ID)
	r.draftKV = lm.NewSliceKV(e.target.Cfg.Layers, e.target.Cfg.DModel)
	r.targetKV = lm.NewSliceKV(e.target.Cfg.Layers, e.target.Cfg.DModel)
	r.draftRNG = lm.NewPCG32(r.Seed, 1)
	r.sampleRNG = lm.NewPCG32(r.Seed, 2)
	e.bm.Grow(r.ID)
	r.state = StatePrefill
}
