package sample

import (
	"math"
	"sort"

	"lmsched/lm"
)

// Params 每个请求独立的采样参数；同一批次里各请求互不影响。
type Params struct {
	Temperature       float64 // 0 表示贪心解码
	TopP              float64 // (0,1]；1.0 表示不做核采样截断
	RepetitionPenalty float64 // 1.0 表示不惩罚
	Greedy            bool
}

// Processor 承载某请求一份处理后的分布，按其私有参数生成。
type Processor struct {
	Params Params
	RNG    *lm.PCG32
}

// Apply 对原始 logits 依次施加：重复惩罚 -> 温度 -> softmax -> 核采样。
// history 是该请求此前已生成（含提示）的 token 序列。
func (p *Processor) Apply(logits []float64, history []int) []float64 {
	if p.Params.Greedy {
		// 贪心不需要分布，但仍施加重复惩罚以保持与采样路径判定口径一致。
		out := append([]float64(nil), logits...)
		applyRepetition(out, history, p.Params.RepetitionPenalty)
		return out
	}
	out := append([]float64(nil), logits...)
	applyRepetition(out, history, p.Params.RepetitionPenalty)

	temp := p.Params.Temperature
	if temp <= 0 {
		temp = 1.0
	}
	for i := range out {
		out[i] /= temp
	}
	probs := softmax(out)
	if p.Params.TopP > 0 && p.Params.TopP < 1.0 {
		probs = topP(probs, p.Params.TopP)
	}
	return probs
}

// Sample 从处理后的分布里采一个 token（贪心取 argmax）。
func (p *Processor) Sample(dist []float64) int {
	if p.Params.Greedy {
		return argmax(dist)
	}
	return Categorical(dist, p.RNG)
}

// Categorical 按分布抽样，用累计概率区间判定，确定性且无第三方依赖。
func Categorical(probs []float64, rng *lm.PCG32) int {
	u := rng.Float64()
	var cum float64
	for i, pr := range probs {
		cum += pr
		if u <= cum { // 闭区间：u=0 且首项概率再小也能正确落到首项
			return i
		}
	}
	// 浮点尾部兜底：选最后一个非零项
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return len(probs) - 1
}

func argmax(x []float64) int {
	best, idx := x[0], 0
	for i := 1; i < len(x); i++ {
		if x[i] > best {
			best, idx = x[i], i
		}
	}
	return idx
}

// AcceptReject 投机解码的接受判定。
// q 是大模型在该位置处理过（温度/核采样/重复惩罚后）的分布，
// p 是草稿模型在同一位置按相同规则处理过的分布，x 是草稿词。
// 返回是否接受。贪心模式下退化为“两边 argmax 相同才接受”，
// 从而保证与逐词贪心解码逐词一致。
func AcceptReject(p, q []float64, x int, greedy bool, rng *lm.PCG32) bool {
	if greedy {
		return argmax(q) == x && argmax(p) == x
	}
	if q[x] <= 0 {
		return false
	}
	ratio := q[x] / p[x]
	if ratio >= 1 {
		return true
	}
	return rng.Float64() < ratio
}

// ResampleNorm 全部草稿被拒后，从修正分布 norm(max(0, q-p)) 重采。
func ResampleNorm(p, q []float64, rng *lm.PCG32) int {
	adj := make([]float64, len(q))
	var sum float64
	for i := range q {
		v := q[i] - p[i]
		if v > 0 {
			adj[i] = v
			sum += v
		}
	}
	if sum == 0 {
		// 数值上 q<=p 处处成立时退回 q（理论上仅浮点误差会走到这里）
		copy(adj, q)
		sum = 0
		for _, v := range adj {
			sum += v
		}
	}
	for i := range adj {
		adj[i] /= sum
	}
	return Categorical(adj, rng)
}

func applyRepetition(logits []float64, history []int, penalty float64) {
	if penalty == 1.0 || penalty <= 0 {
		return
	}
	seen := make(map[int]bool)
	for _, t := range history {
		if seen[t] {
			continue
		}
		seen[t] = true
		if logits[t] > 0 {
			logits[t] /= penalty
		} else {
			logits[t] *= penalty
		}
	}
}

func topP(probs []float64, threshold float64) []float64 {
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })

	var cum float64
	keep := make(map[int]bool)
	for rank, i := range idx {
		cum += probs[i]
		keep[i] = true
		// 至少保留一个 token，避免阈值过小时清空分布
		if cum >= threshold && rank >= 1 {
			break
		}
	}
	out := make([]float64, len(probs))
	var sum float64
	for i, pr := range probs {
		if keep[i] {
			out[i] = pr
			sum += pr
		}
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func softmax(x []float64) []float64 {
	max := math.Inf(-1)
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
