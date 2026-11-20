package engine

import "lmsched/lm"

// runPrefill 把本 tick 的预填充预算按轮转方式分给所有预填充请求，
// 每个请求最多拿 PrefillChunkSize 个 token，长提示被切成小段，
// 与解码请求在 tick 维度交替推进。
func (e *Engine) runPrefill(reqs []*Request) {
	// 轮转起点每 tick 平移，避免固定顺序让某个长请求长期占先。
	start := int(e.roundRobin % uint64(len(reqs)))
	e.roundRobin++
	order := make([]*Request, 0, len(reqs))
	for i := 0; i < len(reqs); i++ {
		order = append(order, reqs[(start+i)%len(reqs)])
	}
	for _, r := range order {
		remaining := len(r.Prompt) - r.prefillPos
		if remaining <= 0 {
			continue // 已完成预填充，无需再处理
		}
		take := e.cfg.PrefillChunkSize
		if take > remaining {
			take = remaining
		}
		e.prefillChunk(r, take)
	}
}

func (e *Engine) prefillChunk(r *Request, take int) {
	start := r.prefillPos
	end := start + take
	tokens := r.Prompt[start:end]
	positions := make([]int, take)
	for i := range positions {
		positions[i] = start + i
	}

	newLen := r.kv.Length + take
	if !e.growIfNeeded(r, newLen) {
		// 抢占失败（极端配置下没有可换出对象）：本 tick 跳过，下 tick 重试。
		return
	}
	resolver := &lm.Resolver{BM: e.bm, KV: r.kv}
	// 因果掩码：新 query 能看到历史 + 本段内不超过自己的位置。
	mask := e.causalMask(resolver.Len(), take)
	out := e.target.Forward([]lm.Segment{{
		Tokens:    tokens,
		Positions: positions,
		ReadKV:    resolver,
		WriteSink: resolver,
		Mask:      mask,
	}})
	e.chargeTarget(r, out.Stats)
	r.kv.Length = newLen
	r.prefillPos = end

	if r.prefillPos >= len(r.Prompt) {
		// 预填充完成：最后一个位置的 logits 预测第一个新 token。
		r.lastTargetLogits = out.Logits[len(out.Logits)-1]
		r.state = StateDecoding
	}
}

// causalMask 生成“KV 视图已有 oldLen 项历史 + n 个新 query”的因果掩码。
func (e *Engine) causalMask(oldLen, n int) [][]bool {
	mask := make([][]bool, n)
	for i := 0; i < n; i++ {
		row := make([]bool, oldLen+n)
		for j := 0; j < oldLen+i+1; j++ {
			row[j] = true
		}
		mask[i] = row
	}
	return mask
}

func (e *Engine) chargeTarget(r *Request, s lm.FwdStats) {
	r.targetForwards++
	r.targetUnits += int64(s.ForwardUnits)
	e.totalTargetFwd++
	e.totalTargetUnits += int64(s.ForwardUnits)
}

func (e *Engine) chargeDraft(r *Request, s lm.FwdStats) {
	r.draftForwards++
	r.draftUnits += int64(s.ForwardUnits)
	e.totalDraftFwd++
	e.totalDraftUnits += int64(s.ForwardUnits)
}
