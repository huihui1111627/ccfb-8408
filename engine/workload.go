package engine

import (
	"lmsched/lm"
	"lmsched/sample"
)

// WorkloadSpec 合成工作负载的形状参数。
type WorkloadSpec struct {
	Requests    int
	Seed        uint64
	ShortPrompt [2]int // 短请求提示长度区间 [lo,hi)
	LongPrompt  [2]int // 长请求提示长度区间
	ShortOut    [2]int
	LongOut     [2]int
	LongRatio   float64 // 长请求占比
	BatchBurst  int     // 每 tick 到达的请求数（1 = 全量一次性到达由调用方控制）
	GreedyRatio float64 // 贪心请求占比（其余按随机采样参数生成）
}

// WorkloadRequest 生成器输出的单个请求描述。
type WorkloadRequest struct {
	Prompt   []int
	MaxNew   int
	Priority int
	Params   sample.Params
	Seed     uint64
	Long     bool
}

// GenerateWorkload 生成长短混杂的请求集。全部随机性走独立 PCG32，
// 同 seed 下逐字节复现，便于不同调度配置间对照。
func GenerateWorkload(cfg lm.Config, spec WorkloadSpec) []WorkloadRequest {
	rng := lm.NewPCG32(spec.Seed, 4242)
	out := make([]WorkloadRequest, 0, spec.Requests)
	for i := 0; i < spec.Requests; i++ {
		long := rng.Float64() < spec.LongRatio
		var pLo, pHi, oLo, oHi int
		if long {
			pLo, pHi = spec.LongPrompt[0], spec.LongPrompt[1]
			oLo, oHi = spec.LongOut[0], spec.LongOut[1]
		} else {
			pLo, pHi = spec.ShortPrompt[0], spec.ShortPrompt[1]
			oLo, oHi = spec.ShortOut[0], spec.ShortOut[1]
		}
		pLen := pLo + int(rng.Uint32()%uint32(pHi-pLo))
		oLen := oLo + int(rng.Uint32()%uint32(oHi-oLo))

		prompt := make([]int, pLen)
		for j := range prompt {
			// 保留 EOS 之外的词表作为提示词
			prompt[j] = int(rng.Uint32() % uint32(cfg.Vocab-1))
		}

		wr := WorkloadRequest{
			Prompt: prompt,
			MaxNew: oLen,
			Long:   long,
			Seed:   spec.Seed + uint64(i)*1009 + 7,
		}
		if rng.Float64() < spec.GreedyRatio {
			wr.Params = sample.Params{Greedy: true, Temperature: 0, TopP: 1, RepetitionPenalty: 1}
		} else {
			// 逐请求不同的温度 / 核采样阈值 / 重复惩罚
			temp := 0.5 + rng.Float64()*1.0  // [0.5,1.5)
			topP := 0.8 + rng.Float64()*0.19 // [0.8,0.99)
			rep := 1.0 + rng.Float64()*0.3   // [1.0,1.3)
			wr.Params = sample.Params{Temperature: temp, TopP: topP, RepetitionPenalty: rep}
		}
		// 优先级 1..5，长请求默认不优待
		wr.Priority = 1 + int(rng.Uint32()%5)
		out = append(out, wr)
	}
	return out
}

// SubmitAll 把工作负载一次性提交给引擎。
func (e *Engine) SubmitAll(wl []WorkloadRequest) []*Request {
	rs := make([]*Request, len(wl))
	for i, w := range wl {
		rs[i] = e.Submit(w.Prompt, w.MaxNew, w.Priority, w.Params, w.Seed)
	}
	return rs
}
