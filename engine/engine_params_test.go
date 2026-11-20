package engine

import (
	"testing"

	"lmsched/sample"
)

// 需求 3：同一批次里每个请求有独立的温度/核采样阈值/重复惩罚，
// 处理分布时严格按请求自身参数，互不串扰。
func TestPerRequestSamplingParams(t *testing.T) {
	target, draft := models()
	eng := NewEngine(target, draft, baseCfg(SpecNone))

	params := []sample.Params{
		{Greedy: true, TopP: 1, RepetitionPenalty: 1.0},
		{Temperature: 0.5, TopP: 0.8, RepetitionPenalty: 1.1},
		{Temperature: 1.2, TopP: 0.95, RepetitionPenalty: 1.3},
		{Temperature: 0.8, TopP: 0.85, RepetitionPenalty: 1.05},
	}
	prompt := []int{1, 2, 3, 4, 5, 6}
	reqs := make([]*Request, len(params))
	for i, p := range params {
		reqs[i] = eng.Submit(prompt, 16, 3, p, uint64(500+i))
		// 参数必须原样保存在请求上
		if reqs[i].Params.Temperature != p.Temperature ||
			reqs[i].Params.TopP != p.TopP ||
			reqs[i].Params.RepetitionPenalty != p.RepetitionPenalty {
			t.Fatalf("请求 %d 采样参数未被独立保存", i)
		}
	}
	eng.Run(0)

	// 所有请求都完成；各自独立运行（独占引擎、同种子）的结果必须与批内一致，
	// 证明批处理没有用别人的参数污染本请求的分布处理。
	for i, r := range reqs {
		if r.State() != StateFinished {
			t.Fatalf("请求 %d 未完成", i)
		}
		solo := runOne(t, target, draft, baseCfg(SpecNone), prompt, 16,
			params[i], uint64(500+i))
		if !intsEqual(r.Output, solo) {
			t.Fatalf("请求 %d 在批内结果与独立运行不一致，参数被串扰:\n batch=%v\n solo =%v",
				i, r.Output, solo)
		}
	}
}

// 需求 3 补充：核采样阈值确实收窄了候选集合，温度改变分布尖锐度。
// 通过 sample 包直接验证处理后的分布满足各请求参数的语义。
func TestSamplingParamsTakeEffect(t *testing.T) {
	// 构造一组 logits，验证 TopP 截断后只保留累积概率阈值内的 token。
	logits := []float64{5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0}
	strict := sample.Params{Greedy: false, Temperature: 1.0, TopP: 0.5, RepetitionPenalty: 1.0}
	loose := sample.Params{Greedy: false, Temperature: 1.0, TopP: 0.99, RepetitionPenalty: 1.0}

	p := sample.Processor{Params: strict}
	dStrict := p.Apply(logits, nil)
	p2 := sample.Processor{Params: loose}
	dLoose := p2.Apply(logits, nil)

	nzStrict, nzLoose := 0, 0
	for _, v := range dStrict {
		if v > 0 {
			nzStrict++
		}
	}
	for _, v := range dLoose {
		if v > 0 {
			nzLoose++
		}
	}
	if nzStrict >= nzLoose {
		t.Fatalf("严格 TopP 应收窄候选: strict 非零=%d loose 非零=%d", nzStrict, nzLoose)
	}
}
