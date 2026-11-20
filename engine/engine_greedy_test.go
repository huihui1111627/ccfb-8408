package engine

import (
	"testing"

	"lmsched/lm"
	"lmsched/sample"
)

func models() (*lm.Model, *lm.Model) {
	cfg := lm.Config{Vocab: 40, DModel: 24, Heads: 4, Layers: 2, DFF: 48, EOS: 39}
	target := lm.NewModel(cfg, 2026)
	draftCfg := lm.Config{Vocab: 40, DModel: 12, Heads: 4, Layers: 2, DFF: 24, EOS: 39}
	draft := lm.NewModel(draftCfg, 2027)
	return target, draft
}

func baseCfg(spec SpecMode) Config {
	return Config{
		PrefillChunkSize: 8,
		Spec:             spec,
		KMin:             1,
		KMax:             6,
		TreeBranches:     3,
		BlockSize:        8,
		GPUBlocks:        64,
		CPUBlocks:        64,
		MaxBatch:         32,
	}
}

func greedyParams() sample.Params {
	return sample.Params{Greedy: true, TopP: 1, RepetitionPenalty: 1.1}
}

func sampleParams(seed int) sample.Params {
	return sample.Params{
		Temperature:       0.7 + float64(seed%5)*0.15,
		TopP:              0.85 + float64(seed%3)*0.04,
		RepetitionPenalty: 1.05 + float64(seed%4)*0.05,
	}
}

func runOne(t *testing.T, target, draft *lm.Model, cfg Config, prompt []int, maxNew int, p sample.Params, seed uint64) []int {
	t.Helper()
	eng := NewEngine(target, draft, cfg)
	r := eng.Submit(prompt, maxNew, 3, p, seed)
	eng.Run(0)
	if r.State() != StateFinished {
		t.Fatalf("请求未完成: state=%s", r.State())
	}
	return append([]int(nil), r.Output...)
}

// 需求 8：贪心模式下，逐词解码 / 链式投机 / 树形投机的输出必须逐词一致。
func TestGreedyEquivalence(t *testing.T) {
	target, draft := models()
	prompts := [][]int{
		{1, 2, 3, 4, 5},
		{7, 7, 8, 8, 9, 10, 11},
		{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5},
		{12, 13, 14},
	}
	for pi, prompt := range prompts {
		var ref []int
		for _, mode := range []SpecMode{SpecNone, SpecChain, SpecTree} {
			cfg := baseCfg(mode)
			out := runOne(t, target, draft, cfg, prompt, 24, greedyParams(), uint64(100+pi))
			if ref == nil {
				ref = out
				continue
			}
			if len(out) != len(ref) {
				t.Fatalf("prompt %d mode %s 长度不一致: %d vs %d", pi, mode, len(out), len(ref))
			}
			for i := range out {
				if out[i] != ref[i] {
					t.Fatalf("prompt %d mode %s 位置 %d 词不一致: %d vs %d", pi, mode, i, out[i], ref[i])
				}
			}
		}
	}
}

// 同一模式下不同 k 上限，贪心输出也必须完全一致（k 只影响速度不影响正确性）。
func TestGreedyInvariantAcrossK(t *testing.T) {
	target, draft := models()
	prompt := []int{5, 6, 7, 8, 9}
	var ref []int
	for _, kmax := range []int{1, 2, 4, 6} {
		cfg := baseCfg(SpecChain)
		cfg.KMin = 1
		cfg.KMax = kmax
		out := runOne(t, target, draft, cfg, prompt, 30, greedyParams(), 42)
		if ref == nil {
			ref = out
			continue
		}
		if !intsEqual(out, ref) {
			t.Fatalf("kmax=%d 贪心输出漂移: %v vs %v", kmax, out, ref)
		}
	}
}

// 需求 7：分块预填充的分块大小不影响最终输出。
func TestPrefillChunkInvariance(t *testing.T) {
	target, draft := models()
	prompt := make([]int, 53)
	for i := range prompt {
		prompt[i] = (i*7 + 3) % 38
	}
	var ref []int
	for _, chunk := range []int{1, 3, 7, 8, 16, 64} {
		cfg := baseCfg(SpecNone)
		cfg.PrefillChunkSize = chunk
		out := runOne(t, target, draft, cfg, prompt, 20, greedyParams(), 9)
		if ref == nil {
			ref = out
			continue
		}
		if !intsEqual(out, ref) {
			t.Fatalf("chunk=%d 输出与基准不一致", chunk)
		}
	}
}

// 需求 7 的调度面：长预填充进行中，短请求每个 tick 仍能推进解码，
// 短请求首字延迟不会被长提示拖到预填充全部结束之后。
func TestShortRequestNotBlocked(t *testing.T) {
	target, draft := models()
	cfg := baseCfg(SpecNone)
	cfg.PrefillChunkSize = 4
	cfg.MaxBatch = 8
	eng := NewEngine(target, draft, cfg)

	longPrompt := make([]int, 80)
	shortPrompt := []int{1, 2, 3}
	longR := eng.Submit(longPrompt, 8, 3, greedyParams(), 1)
	shortR := eng.Submit(shortPrompt, 8, 3, greedyParams(), 2)

	// 推进到短请求出第一个字
	for eng.Tick() < 200 && shortR.firstTokenTick == 0 {
		eng.Step()
	}
	if shortR.firstTokenTick == 0 {
		t.Fatal("短请求始终没有出字")
	}
	// 长请求提示 80 / chunk 4 = 20 个预填充 tick；短请求首字应远早于它全部预填充完成
	if shortR.firstTokenTick >= 20 {
		t.Fatalf("短请求首字被长预填充阻塞: ttft tick=%d", shortR.firstTokenTick)
	}
	if longR.State() == StateFinished {
		t.Fatal("对照用的长请求不应已结束")
	}
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
