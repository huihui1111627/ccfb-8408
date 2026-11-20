package engine

import "testing"

// 需求 2：树形验证与逐条链式验证接受的词必须完全一致。
// 两条路径用相同的模型、草稿提案与随机种子，主链接受判定口径相同。
func TestTreeChainAcceptIdentical(t *testing.T) {
	target, draft := models()
	prompts := [][]int{
		{1, 2, 3, 4, 5},
		{9, 8, 7, 6, 5, 4},
		{2, 4, 6, 8, 10, 12, 14},
	}
	for pi, prompt := range prompts {
		chainCfg := baseCfg(SpecChain)
		treeCfg := baseCfg(SpecTree)
		treeCfg.TreeBranches = 3
		chainOut := runOne(t, target, draft, chainCfg, prompt, 24, greedyParams(), uint6200(pi))
		treeOut := runOne(t, target, draft, treeCfg, prompt, 24, greedyParams(), uint6200(pi))
		if !intsEqual(chainOut, treeOut) {
			t.Fatalf("prompt %d 树形与链式接受词不一致:\n chain=%v\n tree =%v",
				pi, chainOut, treeOut)
		}
	}
}

func uint6200(i int) uint64 { return uint64(200 + i) }

// 需求 5：k 按最近接受率自适应调节，落在 [KMin, KMax] 区间内，
// 且接受率长期偏低时 k 不会持续增长、长期满接受时会增长。
func TestAdaptiveK(t *testing.T) {
	target, draft := models()
	cfg := baseCfg(SpecChain)
	cfg.KMin = 1
	cfg.KMax = 5
	cfg.KAdjustWindow = 3
	eng := NewEngine(target, draft, cfg)

	// 长输出请求，产生足够多提案轮次
	wl := GenerateWorkload(target.Cfg, WorkloadSpec{
		Requests: 8, Seed: 88,
		ShortPrompt: [2]int{8, 16}, LongPrompt: [2]int{16, 24},
		ShortOut: [2]int{30, 50}, LongOut: [2]int{40, 60},
		LongRatio: 0.5, GreedyRatio: 0.5,
	})
	reqs := eng.SubmitAll(wl)
	eng.Run(0)

	// 集成层面：所有请求的 k 始终被约束在 [KMin,KMax]，且接受率统计被记录。
	// 调节方向（满接受上调、全拒绝下调）由 TestAdjustKDirection 精确覆盖。
	for _, r := range reqs {
		if r.State() != StateFinished {
			t.Fatalf("请求 %d 未完成", r.ID)
		}
		if r.currentK < cfg.KMin || r.currentK > cfg.KMax {
			t.Fatalf("k 越界: %d", r.currentK)
		}
		if r.acceptanceRate() < 0 || r.acceptanceRate() > 1.0001 {
			t.Fatalf("接受率越界: %f", r.acceptanceRate())
		}
		if r.totalDrafts <= 0 {
			t.Fatal("请求从未产生草稿提案")
		}
	}
}

// 需求 5 补充：固定满接受场景下 k 应上调；持续拒绝场景下 k 应下调。
// 用单元级方式直接驱动 adjustK 验证方向。
func TestAdjustKDirection(t *testing.T) {
	target, draft := models()
	cfg := baseCfg(SpecChain)
	cfg.KMin = 1
	cfg.KMax = 4
	cfg.KAdjustWindow = 2
	eng := NewEngine(target, draft, cfg)
	r := &Request{currentK: 2, kMin: cfg.KMin, kMax: cfg.KMax}

	// 连续满接受（接受数=提案数）应让 k 上调，直到上限
	for i := 0; i < 10; i++ {
		eng.adjustK(r, 3, 3)
	}
	if r.currentK != cfg.KMax {
		t.Fatalf("满接受未把 k 调到上限: %d", r.currentK)
	}

	// 连续全拒绝应让 k 下调，直到下限
	r.currentK = 4
	for i := 0; i < 10; i++ {
		eng.adjustK(r, 3, 0)
	}
	if r.currentK != cfg.KMin {
		t.Fatalf("全拒绝未把 k 降到下限: %d", r.currentK)
	}
}
