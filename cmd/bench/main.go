// 基准测试：在固定种子的合成长短混合负载上，统计不同投机 k 上限与批大小
// 下的吞吐（token/tick）、首字延迟（TTFT）、目标前向次数与换页次数。
// 纯 Go、无外部依赖，结果以表格打印，便于横向对照。
package main

import (
	"flag"
	"fmt"
	"sort"

	"lmsched/engine"
	"lmsched/lm"
)

type benchResult struct {
	name       string
	spec       engine.SpecMode
	kmax       int
	maxBatch   int
	ticks      int64
	tokens     int
	ttftSum    int64
	ttftMax    int64
	targetFwd  int64
	draftFwd   int64
	swapOut    int64
	swapIn     int64
	finished   int
	converged  bool
	acceptRate float64
}

func main() {
	seed := flag.Uint64("seed", 20261121, "模型固定种子")
	requests := flag.Int("requests", 120, "合成请求数")
	flag.Parse()

	// 基准关注的是调度指标（tok/tick、TTFT、前向次数、换页），这些只依赖
	// 调度逻辑与 tick，与模型算力无关；用轻量模型让基准在秒级跑完。
	cfg := lm.Config{Vocab: 32, DModel: 16, Heads: 4, Layers: 2, DFF: 32, EOS: 31}
	target := lm.NewModel(cfg, *seed)
	draftCfg := lm.Config{Vocab: 32, DModel: 8, Heads: 4, Layers: 2, DFF: 16, EOS: 31}
	draft := lm.NewModel(draftCfg, *seed+1)

	wl := engine.GenerateWorkload(cfg, engine.WorkloadSpec{
		Requests:    *requests,
		Seed:        *seed + 99,
		ShortPrompt: [2]int{4, 10},
		LongPrompt:  [2]int{20, 36},
		ShortOut:    [2]int{6, 14},
		LongOut:     [2]int{16, 30},
		LongRatio:   0.3,
		GreedyRatio: 0.5,
	})

	scenarios := []struct {
		name      string
		spec      engine.SpecMode
		kmax      int
		maxBatch  int
		gpuBlocks int // 0 表示按 batch 自动缩放
	}{
		{"plain-b8", engine.SpecNone, 1, 8, 0},
		{"chain-k1-b8", engine.SpecChain, 1, 8, 0},
		{"chain-k3-b8", engine.SpecChain, 3, 8, 0},
		{"chain-k6-b8", engine.SpecChain, 6, 8, 0},
		{"tree-k4-b8", engine.SpecTree, 4, 8, 0},
		{"chain-k3-b4", engine.SpecChain, 3, 4, 0},
		{"chain-k3-b16", engine.SpecChain, 3, 16, 0},
		{"plain-b16", engine.SpecNone, 1, 16, 0},
		// 显存受限场景：GPU 槽偏紧但仍可推进，强制走换出/换回，观察 TTFT 与吞吐代价。
		{"chain-tight-b8", engine.SpecChain, 3, 4, 16},
	}

	results := make([]benchResult, 0, len(scenarios))
	for _, sc := range scenarios {
		r := runScenario(target, draft, cfg, wl, sc.name, sc.spec, sc.kmax, sc.maxBatch, sc.gpuBlocks)
		results = append(results, r)
	}

	printTable(results)
}

func runScenario(target, draft *lm.Model, cfg lm.Config, wl []engine.WorkloadRequest,
	name string, spec engine.SpecMode, kmax, maxBatch, gpuOverride int) benchResult {
	// GPU 槽随批大小缩放：保证每个并发槽位有足够块，避免极端超载导致换页活锁；
	// gpuOverride>0 时用受限配置，真实触发换出/换回。
	gpuBlocks := maxBatch * 6
	if gpuOverride > 0 {
		gpuBlocks = gpuOverride
	}
	ecfg := engine.Config{
		PrefillChunkSize: 8,
		Spec:             spec,
		KMin:             1,
		KMax:             kmax,
		TreeBranches:     3,
		BlockSize:        8,
		GPUBlocks:        gpuBlocks,
		CPUBlocks:        96,
		MaxBatch:         maxBatch,
	}
	eng := engine.NewEngine(target, draft, ecfg)
	reqs := eng.SubmitAll(wl)
	_, converged := eng.RunWithGuard(800)

	res := benchResult{name: name, spec: spec, kmax: kmax, maxBatch: maxBatch,
		ticks: eng.Tick(), finished: len(eng.Finished())}
	res.converged = converged
	var acceptSum float64
	for _, r := range reqs {
		res.tokens += len(r.Output)
		if r.FirstTokenTick() > 0 {
			ttft := r.FirstTokenTick() - r.AdmittedTick()
			res.ttftSum += ttft
			if ttft > res.ttftMax {
				res.ttftMax = ttft
			}
		}
		res.targetFwd += r.TargetForwards()
		res.draftFwd += r.DraftForwards()
		acceptSum += r.AcceptanceRate()
	}
	bm := eng.BlockManager()
	res.swapOut = int64(bm.SwapsOut)
	res.swapIn = int64(bm.SwapsIn)
	if len(reqs) > 0 {
		res.acceptRate = acceptSum / float64(len(reqs))
	}
	return res
}

func printTable(rs []benchResult) {
	sort.SliceStable(rs, func(i, j int) bool { return rs[i].name < rs[j].name })
	fmt.Println("\n=== 投机解码调度基准（固定种子，合成长短混合负载）===")
	header := fmt.Sprintf("%-14s %-6s %-4s %-6s | %-8s %-9s %-9s %-9s | %-9s %-9s %-8s %-7s",
		"scenario", "mode", "kmax", "batch", "tokens", "tok/tick", "TTFT均值", "TTFT最大",
		"目标前向", "草稿前向", "接受率", "换出/入")
	fmt.Println(header)
	for _, r := range rs {
		mode := "none"
		switch r.spec {
		case engine.SpecChain:
			mode = "chain"
		case engine.SpecTree:
			mode = "tree"
		}
		throughput := 0.0
		ttftAvg := 0.0
		if r.ticks > 0 {
			throughput = float64(r.tokens) / float64(r.ticks)
		}
		if r.finished > 0 {
			ttftAvg = float64(r.ttftSum) / float64(r.finished)
		}
		fmt.Printf("%-14s %-6s %-4d %-6d | %-8d %-9.3f %-9.2f %-9d | %-9d %-9d %-8.3f %d/%d%s\n",
			r.name, mode, r.kmax, r.maxBatch,
			r.tokens, throughput, ttftAvg, r.ttftMax,
			r.targetFwd, r.draftFwd, r.acceptRate, r.swapOut, r.swapIn,
			convMark(r.converged))
	}
	fmt.Println()
}

// convMark 标注场景是否完全收敛（所有请求正常完成）。
func convMark(ok bool) string {
	if ok {
		return ""
	}
	return "  [未完全收敛:换页压力过大]"
}
