package engine

import (
	"math"
	"testing"

	"lmsched/lm"
	"lmsched/sample"
)

func newTestPCG(seed uint64) *lm.PCG32 { return lm.NewPCG32(seed, 1) }

// chiSquare 计算卡方统计量：观测计数 vs 期望计数。
func chiSquare(observed []int, probs []float64, n int) float64 {
	stat := 0.0
	for i, o := range observed {
		e := probs[i] * float64(n)
		if e < 1e-9 {
			if o != 0 {
				stat = math.Inf(1)
			}
			continue
		}
		d := float64(o) - e
		stat += d * d / e
	}
	return stat
}

// 需求 4（分布层面）：投机解码的接受-拒绝采样输出分布必须与“仅目标模型
// 采样”一致。用固定的目标/草稿条件分布构造大量独立试验，比较经验频率。
func TestSpeculativeDistributionMatchesTarget(t *testing.T) {
	// 固定一对分布 q（目标）与 p（草稿），覆盖 q>p 与 q<p 两类位置。
	q := []float64{0.35, 0.25, 0.20, 0.15, 0.05}
	p := []float64{0.25, 0.30, 0.20, 0.15, 0.10}
	rng := newTestPCG(12345)
	n := 200000
	observed := make([]int, len(q))
	for i := 0; i < n; i++ {
		// 草稿从 p 采一个候选 x
		x := sample.Categorical(p, rng)
		var tok int
		if sample.AcceptReject(p, q, x, false, rng) {
			tok = x
		} else {
			tok = sample.ResampleNorm(p, q, rng)
		}
		observed[tok]++
	}
	// 投机解码接受-拒绝的边缘分布定理：最终分布恰好等于 q。
	stat := chiSquare(observed, q, n)
	// 自由度 4、显著性 0.01 的卡方临界值约 13.28
	if stat > 13.28 {
		freq := make([]float64, len(q))
		for i := range freq {
			freq[i] = float64(observed[i]) / float64(n)
		}
		t.Fatalf("投机采样分布偏离目标 q: chi2=%.3f freq=%v", stat, freq)
	}
}

// 需求 4（端到端）：在一个固定的真实模型上下文上，反复跑“草稿提案 + 目标
// 验证”的完整投机流程，第一个新词的边缘分布必须等于目标模型在该上下文上
// 直接采样的分布。固定上下文消除了混合分布干扰，是严格的分布等价检验。
func TestEndToEndSamplingDistribution(t *testing.T) {
	target, draft := models()
	prompt := []int{1, 2, 3, 4, 5}
	params := sample.Params{Temperature: 0.9, TopP: 0.95, RepetitionPenalty: 1.1}
	const trials = 4000

	firstPlain := make([]int, trials)
	firstChain := make([]int, trials)
	for i := 0; i < trials; i++ {
		// 每次试验用独立种子，两条路径各取一条独立随机流
		firstPlain[i] = runOne(t, target, draft, baseCfg(SpecNone),
			prompt, 1, params, uint64(30000+i))[0]
		firstChain[i] = runOne(t, target, draft, baseCfg(SpecChain),
			prompt, 1, params, uint64(80000+i))[0]
	}

	vocab := target.Cfg.Vocab
	cP := make([]int, vocab)
	cC := make([]int, vocab)
	for i := 0; i < trials; i++ {
		cP[firstPlain[i]]++
		cC[firstChain[i]]++
	}
	// 以纯目标经验分布为期望（拉普拉斯平滑），检验链式边缘分布。
	probs := make([]float64, vocab)
	for v := range probs {
		probs[v] = (float64(cP[v]) + 0.5) / (float64(trials) + 0.5*float64(vocab))
	}
	stat := chiSquare(cC, probs, trials)
	threshold := 2.0 * float64(vocab-1) // 宽松阈值，捕捉系统性塌陷
	if stat > threshold {
		t.Fatalf("固定上下文上链式首词边缘分布偏离纯目标: chi2=%.2f threshold=%.2f",
			stat, threshold)
	}

	// 额外保证：两条路径用到的 token 种类都应较丰富，没有退化到单一输出。
	distinct := map[int]bool{}
	for _, tok := range firstChain {
		distinct[tok] = true
	}
	if len(distinct) < vocab/4 {
		t.Fatalf("链式采样多样性塌陷: 仅 %d 种 token", len(distinct))
	}
}
