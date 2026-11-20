package sample

import (
	"math"
	"testing"

	"lmsched/lm"
)

// 核采样：截断后概率质量只保留最小的 top-p 覆盖集合，且重新归一。
func TestTopP(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.15, 0.04, 0.01}
	out := topP(probs, 0.9)
	if out[3] != 0 || out[4] != 0 {
		t.Fatalf("top-p 未截断尾部: %v", out)
	}
	var sum float64
	for _, v := range out {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("top-p 后未归一: %v", sum)
	}
	// 至少保留一个
	out2 := topP(probs, 0.01)
	nz := 0
	for _, v := range out2 {
		if v > 0 {
			nz++
		}
	}
	if nz < 1 {
		t.Fatal("top-p 阈值过小导致空分布")
	}
}

// 重复惩罚：正 logits 除以惩罚，负 logits 乘以惩罚。
func TestRepetitionPenalty(t *testing.T) {
	logits := []float64{2.0, -2.0, 1.0}
	applyRepetition(logits, []int{0, 1}, 1.5)
	if math.Abs(logits[0]-(2.0/1.5)) > 1e-12 {
		t.Fatal("正 logits 惩罚错误")
	}
	if math.Abs(logits[1]-(-3.0)) > 1e-12 {
		t.Fatal("负 logits 惩罚错误")
	}
	if logits[2] != 1.0 {
		t.Fatal("未出现词不应被惩罚")
	}
}

// 同种子类别采样逐位复现；大量试验的经验频率收敛到给定分布。
func TestCategoricalConverges(t *testing.T) {
	rng := lm.NewPCG32(7, 1)
	probs := []float64{0.1, 0.2, 0.3, 0.4}
	counts := make([]int, len(probs))
	n := 200000
	for i := 0; i < n; i++ {
		counts[Categorical(probs, rng)]++
	}
	for i, p := range probs {
		got := float64(counts[i]) / float64(n)
		if math.Abs(got-p) > 0.01 {
			t.Fatalf("类别采样频率偏离: token=%d got=%.4f want=%.4f", i, got, p)
		}
	}
}

// 贪心 Processor 永远取 argmax，且与 RNG 无关。
func TestGreedyArgmax(t *testing.T) {
	p := &Processor{Params: Params{Greedy: true}, RNG: lm.NewPCG32(1, 1)}
	logits := []float64{0.1, 3.0, 2.0, -1.0}
	dist := p.Apply(logits, nil)
	if p.Sample(dist) != 1 {
		t.Fatal("贪心未取 argmax")
	}
}

// 拒绝采样：q(x) >= p(x) 时必接受；q(x)=0 时必拒绝。
func TestAcceptRejectBoundaries(t *testing.T) {
	rng := lm.NewPCG32(2, 2)
	if !AcceptReject([]float64{0.2, 0.8}, []float64{0.5, 0.5}, 0, false, rng) {
		t.Fatal("q>=p 应当必接受")
	}
	if AcceptReject([]float64{0.5, 0.5}, []float64{0.0, 1.0}, 0, false, rng) {
		t.Fatal("q(x)=0 应当必拒绝")
	}
	// 贪心：两边 argmax 一致才接受
	if AcceptReject([]float64{0.9, 0.1}, []float64{0.6, 0.4}, 1, true, rng) {
		t.Fatal("贪心模式 argmax 不一致却接受了")
	}
	if !AcceptReject([]float64{0.9, 0.1}, []float64{0.8, 0.2}, 0, true, rng) {
		t.Fatal("贪心模式双方 argmax 相同却拒绝了")
	}
}

// 修正分布归一且只在 q>p 的 token 上有质量。
func TestResampleNorm(t *testing.T) {
	p := []float64{0.5, 0.3, 0.2}
	q := []float64{0.2, 0.5, 0.3}
	rng := lm.NewPCG32(3, 3)
	tok := ResampleNorm(p, q, rng)
	if tok == 0 {
		t.Fatal("修正分布在 q<=p 的 token 上不应有质量")
	}
}
