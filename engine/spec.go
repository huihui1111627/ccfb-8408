package engine

import (
	"lmsched/lm"
	"lmsched/sample"
)

// draftProposal 一次草稿提案。tokens/draftRaw 描述主链（树模式下即分支 0），
// branches 仅树模式使用：每条分支是 b 个独立候选词，接在主链第 1 层之后。
type draftProposal struct {
	tokens        []int
	draftRaw      [][]float64
	targetBaseRaw []float64

	branchTokens [][]int // [b-1][k-1]，分支 b（b>=1）的词序列
	branchRaw    [][][]float64
}

// proposeChain 草稿模型顺序猜 k 个词，瞬态 KV 与分页前缀对齐。
func (e *Engine) proposeChain(r *Request, k int) *draftProposal {
	p := &draftProposal{targetBaseRaw: cloneVec(r.lastTargetLogits)}
	// 瞬态 KV 每轮提案都从空开始：它只承载本轮草稿/目标的 k 个临时位置，
	// 已提交前缀始终在分页块里。Trim 用 0 才能真正清空上一轮残留。
	r.draftKV.Trim(0)
	r.targetKV.Trim(0)
	history := r.History()

	for t := 0; t < k; t++ {
		var inTok int
		if t == 0 {
			inTok = history[len(history)-1]
		} else {
			inTok = p.tokens[t-1]
		}
		out := e.draftForward(r, []int{inTok}, r.kv.Length+t-1)
		raw := out.Logits[len(out.Logits)-1]
		hist := append(append([]int(nil), history...), p.tokens[:t]...)
		proc := &sample.Processor{Params: r.Params, RNG: r.draftRNG}
		dist := proc.Apply(raw, hist)
		tok := proc.Sample(dist)
		p.tokens = append(p.tokens, tok)
		p.draftRaw = append(p.draftRaw, raw)
	}
	return p
}

// draftForward 在“分页前缀 + 瞬态草稿 KV”上跑一次草稿前向并写瞬态 KV。
func (e *Engine) draftForward(r *Request, toks []int, startPos int) *lm.Output {
	n := len(toks)
	// 写槽（草稿瞬态 KV）的扩容由 Model.Forward 内部的 WriteBase 统一负责，
	// 这里不能预扩，否则读视图长度会与掩码行数错位。
	suffixBase := r.draftKV.Len()
	comp := &lm.CompositeKV{
		Prefix: &lm.Resolver{BM: e.bm, KV: r.kv},
		Suffix: r.draftKV,
	}
	// 掩码按“写入后”的视图长度分配（前缀 + 旧后缀 + 本段 n），
	// 与模型内 WriteBase 扩容后的 ReadKV.Len() 严格相等。
	viewLen := r.kv.Length + suffixBase + n
	positions := make([]int, n)
	for i := range positions {
		positions[i] = startPos + i
	}
	mask := make([][]bool, n)
	for i := 0; i < n; i++ {
		row := make([]bool, viewLen)
		// query i 可见到全局位置 前缀长+suffixBase+i（含自身），其后的未写槽全遮。
		limit := r.kv.Length + suffixBase + i + 1
		for j := 0; j < limit; j++ {
			row[j] = true
		}
		mask[i] = row
	}
	out := e.draft.Forward([]lm.Segment{{
		Tokens:    toks,
		Positions: positions,
		ReadKV:    comp,
		WriteSink: r.draftKV, // 相对下标写入
		Mask:      mask,
	}})
	e.chargeDraft(r, out.Stats)
	return out
}

// verifyResult 一次验证的结构化结果（链式/树形共用）。
type verifyResult struct {
	accepted    []int
	targetRaw   [][]float64 // 主链每个提案位置的目标原始 logits
	replacement int
	hasReplace  bool
	bonus       int
	hasBonus    bool
	longestPath int // 树模式：所有分支里最长的接受路径长度（统计/展示用）
}

// runChain 链式投机解码。
func (e *Engine) runChain(r *Request) {
	k := r.currentK
	prop := e.proposeChain(r, k)
	res := e.verifyChain(r, prop)
	res.longestPath = len(res.accepted)
	e.verifyAndCommit(r, prop, res)
	e.adjustK(r, len(prop.tokens), len(res.accepted))
}

// verifyChain 目标逐条验证，严格按投机解码的位置对齐：
//   - targetBaseRaw 是目标在“已提交序列”上的输出，预测第 L 个位置（首个新词）
//   - draftRaw[t] 与“上一个被接受词的目标前向输出”都预测第 L+t 个位置，
//     用这一对分布判定草稿词 x_t 接受/拒绝
//   - 接受后才对 x_t 跑目标前向：写目标自己的 K/V，并产出预测 L+t+1 的分布
func (e *Engine) verifyChain(r *Request, prop *draftProposal) *verifyResult {
	res := &verifyResult{}
	history := r.History()
	r.targetKV.Trim(0)

	// currentRaw：目标对“当前待判定位置”的原始 logits。
	// 初始即基线（预测位置 L）；每接受一个词后由目标前向刷新到下一位置。
	currentRaw := prop.targetBaseRaw

	for t, x := range prop.tokens {
		hist := append(append([]int(nil), history...), prop.tokens[:t]...)
		proc := &sample.Processor{Params: r.Params, RNG: r.sampleRNG}
		q := proc.Apply(currentRaw, hist)       // 目标分布，预测位置 L+t
		p := proc.Apply(prop.draftRaw[t], hist) // 草稿分布，预测同一位置

		if !sample.AcceptReject(p, q, x, r.Params.Greedy, r.sampleRNG) {
			// 拒绝：修正分布重采（贪心取目标 argmax）。targetKV 此刻恰含 t 个已接受词。
			if !r.Params.Greedy {
				res.replacement = sample.ResampleNorm(p, q, r.sampleRNG)
			} else {
				res.replacement = argmaxOf(q)
			}
			res.hasReplace = true
			return res
		}

		// 接受 x_t：跑目标前向，把目标 K/V 写进 targetKV，并取预测下一位置的 logits。
		res.accepted = append(res.accepted, x)
		currentRaw = e.targetForwardOne(r, x, r.kv.Length+t, t)
	}

	// 全部接受：用最终目标分布采奖励词。
	proc := &sample.Processor{Params: r.Params, RNG: r.sampleRNG}
	hist := append(append([]int(nil), history...), prop.tokens...)
	q := proc.Apply(currentRaw, hist)
	res.bonus = proc.Sample(q)
	res.hasBonus = true
	return res
}

// targetForwardOne 在“分页前缀 + 已接受目标瞬态 KV”上对单 token 跑目标前向。
// 扩容由 Forward 内部 WriteBase 负责；掩码按写入后视图长度分配，本位置因果可见全部。
func (e *Engine) targetForwardOne(r *Request, tok, pos, wantSuffix int) []float64 {
	suffixBase := r.targetKV.Len() // 调用前恰有 wantSuffix 项
	comp := &lm.CompositeKV{
		Prefix: &lm.Resolver{BM: e.bm, KV: r.kv},
		Suffix: r.targetKV,
	}
	viewLen := r.kv.Length + suffixBase + 1
	mask := [][]bool{make([]bool, viewLen)}
	for j := 0; j < viewLen; j++ {
		mask[0][j] = true
	}
	out := e.target.Forward([]lm.Segment{{
		Tokens:    []int{tok},
		Positions: []int{pos},
		ReadKV:    comp,
		WriteSink: r.targetKV,
		Mask:      mask,
	}})
	e.chargeTarget(r, out.Stats)
	return out.Logits[0]
}

// verifyAndCommit 把验证结果落到分页 KV 与请求输出。
// 落盘只走主链，因此输出分布与“仅目标模型采样”严格一致。
func (e *Engine) verifyAndCommit(r *Request, prop *draftProposal, res *verifyResult) {
	r.totalDrafts += int64(len(prop.tokens))
	r.totalAccepted += int64(len(res.accepted))
	r.treeLongest = res.longestPath

	nAcc := len(res.accepted)
	base := r.kv.Length
	if !e.growIfNeeded(r, base+nAcc+1) {
		return
	}
	// 1) 接受路径的【目标模型】KV 搬到分页块（不重新计算前缀）。
	// targetKV 相对下标 0..nAcc-1 对应逻辑位置 base..base+nAcc-1。
	resolver := &lm.Resolver{BM: e.bm, KV: r.kv}
	for j := 0; j < nAcc; j++ {
		logical := base + j
		for l := 0; l < e.target.Cfg.Layers; l++ {
			resolver.WriteK(l, logical, r.targetKV.K(l, j))
			resolver.WriteV(l, logical, r.targetKV.V(l, j))
		}
	}
	r.kv.Length = base + nAcc

	// 2) 提交接受词
	e.emit(r, res.accepted)
	if r.state == StateFinished {
		r.draftKV.Trim(0)
		r.targetKV.Trim(0)
		return
	}

	// 3) 替换词/奖励词
	var nextTok int
	if res.hasReplace {
		nextTok = res.replacement
	} else if res.hasBonus {
		nextTok = res.bonus
	} else {
		r.draftKV.Trim(0)
		r.targetKV.Trim(0)
		return
	}
	e.emit(r, []int{nextTok})
	if r.state == StateFinished {
		r.kv.Length++
		r.draftKV.Trim(0)
		r.targetKV.Trim(0)
		return
	}

	// 4) 目标前向落盘替换/奖励词，并准备下一 tick 的基线 logits。
	e.advanceTargetOne(r, nextTok)
	r.draftKV.Trim(0)
	r.targetKV.Trim(0)
}

// adjustK 用最近提案的接受率更新 EMA，并按窗口自适应 k。
func (e *Engine) adjustK(r *Request, drafts, accepted int) {
	if drafts == 0 {
		return
	}
	rate := float64(accepted) / float64(drafts)
	if r.totalDrafts-int64(drafts) == 0 {
		r.acceptEMA = rate
	} else {
		a := e.cfg.EmaAlpha
		r.acceptEMA = a*rate + (1-a)*r.acceptEMA
	}
	r.emittedSinceKAdjust++
	if r.emittedSinceKAdjust < e.cfg.KAdjustWindow {
		return
	}
	r.emittedSinceKAdjust = 0
	switch {
	case r.acceptEMA >= 0.7 && r.currentK < r.kMax:
		r.currentK++
	case r.acceptEMA <= 0.3 && r.currentK > r.kMin:
		r.currentK--
	}
}

func argmaxOf(x []float64) int {
	best, idx := x[0], 0
	for i := 1; i < len(x); i++ {
		if x[i] > best {
			best, idx = x[i], i
		}
	}
	return idx
}

func cloneVec(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}
