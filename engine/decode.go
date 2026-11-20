package engine

import (
	"lmsched/lm"
	"lmsched/sample"
)

// runDecode 一个解码 tick：按配置走逐词/链式/树形路径。
func (e *Engine) runDecode(r *Request) {
	switch e.cfg.Spec {
	case SpecChain:
		e.runChain(r)
	case SpecTree:
		e.runTree(r)
	default:
		e.runPlain(r)
	}
}

// runPlain 普通逐词解码：用 lastTargetLogits 出一个词，再跑一次目标前向
// 为下一个位置准备 logits 并写入分页 KV。
func (e *Engine) runPlain(r *Request) {
	proc := &sample.Processor{Params: r.Params, RNG: r.sampleRNG}
	dist := proc.Apply(r.lastTargetLogits, r.History())
	tok := proc.Sample(dist)
	e.emit(r, []int{tok})
	if r.state == StateFinished {
		return
	}
	e.advanceTargetOne(r, tok)
}

// advanceTargetOne 对单 token 做目标前向，写分页 KV，并更新 lastTargetLogits。
func (e *Engine) advanceTargetOne(r *Request, tok int) {
	pos := r.kv.Length
	if !e.growIfNeeded(r, pos+1) {
		return
	}
	resolver := &lm.Resolver{BM: e.bm, KV: r.kv}
	// 分页视图含全部 pos 个历史 + 本位置；单 query 因果可见全部，必须显式传掩码
	// （nil 掩码只适用于“读视图恰好等于本段”的纯段前向）。
	viewLen := pos + 1
	mask := [][]bool{make([]bool, viewLen)}
	for j := range mask[0] {
		mask[0][j] = true
	}
	out := e.target.Forward([]lm.Segment{{
		Tokens:    []int{tok},
		Positions: []int{pos},
		ReadKV:    resolver,
		WriteSink: resolver,
		Mask:      mask,
	}})
	e.chargeTarget(r, out.Stats)
	r.kv.Length = pos + 1
	r.lastTargetLogits = out.Logits[0]
}

// emit 提交一批新生成的 token，处理 EOS、长度上限和指标。
func (e *Engine) emit(r *Request, toks []int) {
	for _, tok := range toks {
		r.Output = append(r.Output, tok)
		if r.firstTokenTick == 0 {
			r.firstTokenTick = e.tick
		}
		r.recordEmit(e.tick, 1)
		if tok == e.target.Cfg.EOS {
			r.state = StateFinished
			r.finishReason = "eos"
			return
		}
		if len(r.Output) >= r.MaxNewTokens {
			r.state = StateFinished
			r.finishReason = "length"
			return
		}
	}
}
