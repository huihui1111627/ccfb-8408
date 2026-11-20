package engine

import (
	"lmsched/lm"
	"lmsched/sample"
)

// treeBranch 树里的一条候选分支。分支 0 是主链（与链式验证同构的路径），
// 其余分支与主链共享第 0 层候选，从第 1 层开始分叉。
type treeBranch struct {
	tokens []int // 长度 k
	raw    [][]float64
	kv     *lm.SliceKV
}

// proposeTree 构造一棵 B 分支、深度 k 的候选树。
// 各分支第 0 层共享同一个候选词（根候选），其后独立采样、独立瞬态 KV。
func (e *Engine) proposeTree(r *Request, k, branches int) ([]*treeBranch, []float64) {
	history := r.History()
	baseRaw := cloneVec(r.lastTargetLogits)
	out := make([]*treeBranch, branches)

	for b := 0; b < branches; b++ {
		br := &treeBranch{kv: lm.NewSliceKV(e.target.Cfg.Layers, e.target.Cfg.DModel)}
		for t := 0; t < k; t++ {
			var inTok int
			if t == 0 {
				inTok = history[len(history)-1]
			} else if b == 0 {
				inTok = br.tokens[t-1]
			} else {
				inTok = br.tokens[t-1]
			}
			// 每个分支用自己独立的瞬态 KV；第 0 层输入相同。
			// 扩容交给 Forward 内部 WriteBase；掩码按写入后视图长度分配。
			suffixBase := br.kv.Len()
			comp := &lm.CompositeKV{
				Prefix: &lm.Resolver{BM: e.bm, KV: r.kv},
				Suffix: br.kv,
			}
			viewLen := r.kv.Length + suffixBase + 1
			pos := r.kv.Length + t - 1
			mask := [][]bool{fullVisibleMask(viewLen)}
			fwd := e.draft.Forward([]lm.Segment{{
				Tokens:    []int{inTok},
				Positions: []int{pos},
				ReadKV:    comp,
				WriteSink: br.kv,
				Mask:      mask,
			}})
			e.chargeDraft(r, fwd.Stats)
			raw := fwd.Logits[0]
			hist := append(append([]int(nil), history...), br.tokens[:t]...)
			// 分支 0 用主草稿流；侧分支用独立子流，保证主链 RNG 序列与链式完全一致。
			rng := r.draftRNG
			if b > 0 {
				rng = lm.NewPCG32(r.Seed, uint64(100+b*13+t))
			}
			proc := &sample.Processor{Params: r.Params, RNG: rng}
			dist := proc.Apply(raw, hist)
			tok := proc.Sample(dist)
			br.tokens = append(br.tokens, tok)
			br.raw = append(br.raw, raw)
		}
		out[b] = br
	}
	return out, baseRaw
}

func fullVisibleMask(n int) []bool {
	m := make([]bool, n)
	for i := range m {
		m[i] = true
	}
	return m
}

// runTree 树形投机解码：一次带树形掩码的目标前向验完整棵树。
func (e *Engine) runTree(r *Request) {
	k := r.currentK
	B := e.cfg.TreeBranches
	branches, baseRaw := e.proposeTree(r, k, B)

	// 组装一次前向的多个 Segment：每分支一段，前缀都是分页 KV，
	// 段内掩码只允许沿该分支自身路径可见（分支间互不可见）。
	segs := make([]lm.Segment, B)
	for b, br := range branches {
		comp := &lm.CompositeKV{
			Prefix: &lm.Resolver{BM: e.bm, KV: r.kv},
			Suffix: br.kv,
		}
		prefixLen := r.kv.Length
		n := len(br.tokens)
		mask := make([][]bool, n)
		for i := 0; i < n; i++ {
			row := make([]bool, comp.Len())
			for j := 0; j < prefixLen+i+1; j++ {
				row[j] = true
			}
			mask[i] = row
		}
		positions := make([]int, n)
		for i := range positions {
			positions[i] = r.kv.Length + i
		}
		segs[b] = lm.Segment{
			Tokens:    br.tokens,
			Positions: positions,
			ReadKV:    comp,
			Mask:      mask,
		}
	}
	fwd := e.target.Forward(segs)
	e.chargeTarget(r, fwd.Stats)

	history := r.History()
	main := branches[0]

	// 侧分支用树形前向的 logits 独立统计接受路径长度（不参与出字、不写 KV），
	// 取最长用于展示和 k 调节。
	longest := 0
	for b := 1; b < B; b++ {
		br := branches[b]
		off := b * k
		length := 0
		for t := 0; t < k; t++ {
			raw := fwd.Logits[off+t]
			hist := append(append([]int(nil), history...), br.tokens[:t]...)
			proc := &sample.Processor{Params: r.Params, RNG: lm.NewPCG32(r.Seed, uint64(200+b*17+t))}
			q := proc.Apply(raw, hist)
			p := proc.Apply(br.raw[t], hist)
			if !sample.AcceptReject(p, q, br.tokens[t], r.Params.Greedy, proc.RNG) {
				break
			}
			length++
		}
		if length > longest {
			longest = length
		}
	}

	// 主链（分支 0）的接受判定、目标 K/V 构建与落盘完全复用链式逻辑，
	// 从而保证树形验证与逐条链式验证接受的词完全一致。
	r.draftKV = main.kv
	prop := &draftProposal{
		tokens:        main.tokens,
		draftRaw:      main.raw,
		targetBaseRaw: baseRaw,
	}
	res := e.verifyChain(r, prop)
	if longest < len(res.accepted) {
		longest = len(res.accepted)
	}
	res.longestPath = longest
	e.verifyAndCommit(r, prop, res)
	e.adjustK(r, len(prop.tokens), longest)
}
