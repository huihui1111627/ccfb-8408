package engine

import (
	"testing"
)

// 需求 6：显存块不够时整个请求换出到 CPU，换回后接着解码，
// 不重算前缀，输出与从未换页的运行完全一致。
func TestSwapPreservesOutput(t *testing.T) {
	target, draft := models()

	// 对照配置：显存充足，不发生换页
	cfgFull := baseCfg(SpecChain)
	cfgFull.GPUBlocks = 64

	// 紧张但可推进的配置：GPU 槽只够同时容纳约一半请求，
	// 必然触发换出/换回，但不至于极端超载导致持续颠簸。
	cfgTight := baseCfg(SpecChain)
	cfgTight.BlockSize = 8
	// GPU 槽只够 2 个大请求，但允许 4 个并发上卡，预填充增长时必然抢占换出。
	cfgTight.GPUBlocks = 8
	cfgTight.CPUBlocks = 64
	cfgTight.MaxBatch = 4
	eng := NewEngine(target, draft, cfgTight)

	wl := GenerateWorkload(target.Cfg, WorkloadSpec{
		Requests: 10, Seed: 555,
		ShortPrompt: [2]int{4, 8}, LongPrompt: [2]int{12, 20},
		ShortOut: [2]int{6, 12}, LongOut: [2]int{10, 18},
		LongRatio: 0.4, GreedyRatio: 1.0,
	})
	reqs := eng.SubmitAll(wl)
	eng.Run(0)

	// 必须真的发生过换页，否则测试没有覆盖目标路径
	bm := eng.BlockManager()
	if bm.SwapsOut == 0 || bm.SwapsIn == 0 {
		t.Fatalf("未触发换出/换回: out=%d in=%d", bm.SwapsOut, bm.SwapsIn)
	}

	// 所有请求都完成，且换过页的请求输出与其在充足显存下独立运行的结果一致
	swapped := 0
	for i, r := range reqs {
		if r.State() != StateFinished {
			t.Fatalf("请求 %d 未完成", r.ID)
		}
		if !r.everSwapped {
			continue
		}
		swapped++
		sole := runOne(t, target, draft, cfgFull, wl[i].Prompt, wl[i].MaxNew,
			wl[i].Params, wl[i].Seed)
		if !intsEqual(r.Output, sole) {
			t.Fatalf("换页请求 #%d 输出漂移:\n swapped=%v\n sole=%v", r.ID, r.Output, sole)
		}
	}
	if swapped == 0 {
		t.Fatal("没有请求经历过换页")
	}
}

// 换出/换回的结构性保证：逻辑长度、块数量与 K/V 内容不变，物理槽可被复用
// （换回时块表允许改写到新的 GPU 槽，但每个逻辑块仍能读到原来的数据）。
func TestSwapKeepsBlockTable(t *testing.T) {
	target, draft := models()
	cfg := baseCfg(SpecNone)
	cfg.GPUBlocks = 8
	cfg.CPUBlocks = 16
	cfg.MaxBatch = 2
	eng := NewEngine(target, draft, cfg)
	highPrio := eng.Submit(make([]int, 20), 10, 5, greedyParams(), 1)
	victim := eng.Submit(make([]int, 20), 10, 1, greedyParams(), 2)
	if highPrio.ID == victim.ID {
		t.Fatal("请求 ID 冲突")
	}

	// 推进到两个请求都上卡并占用块
	for eng.Tick() < 12 {
		eng.Step()
	}
	bm := eng.BlockManager()
	lenBefore := victim.kv.Length
	blocksBefore := len(victim.kv.Table)

	// 记录换出前每个逻辑位置的 K/V 快照
	snapK := make([][]float64, lenBefore)
	for j := 0; j < lenBefore; j++ {
		phys := victim.kv.Table[j/cfg.BlockSize]
		snapK[j] = append([]float64(nil), bm.ReadK(phys, 0, j%cfg.BlockSize)...)
	}

	eng.bm.SwapOut(victim.ID)
	if len(victim.kv.Table) != blocksBefore {
		t.Fatal("换出改变了块数量")
	}
	if victim.kv.Length != lenBefore {
		t.Fatal("换出改变了逻辑长度")
	}
	if bm.SwapIn(victim.ID) != blocksBefore {
		t.Fatal("换回块数与换出不一致")
	}

	// 换回后逐位置核对 K/V 内容不变（即使物理槽 ID 已改写）
	for j := 0; j < lenBefore; j++ {
		phys := victim.kv.Table[j/cfg.BlockSize]
		got := bm.ReadK(phys, 0, j%cfg.BlockSize)
		for d := range got {
			if got[d] != snapK[j][d] {
				t.Fatalf("位置 %d 换页后 K/V 内容漂移", j)
			}
		}
	}
	if victim.kv.Length != lenBefore {
		t.Fatal("换回改变了逻辑长度")
	}
}

// 需求 1：连续批处理——每个 tick 都有请求结束并被回收、等待请求被换入，
// 批次内容动态变化而不是等整批结束。
func TestContinuousBatching(t *testing.T) {
	target, draft := models()
	cfg := baseCfg(SpecNone)
	cfg.MaxBatch = 4
	eng := NewEngine(target, draft, cfg)

	// 短请求先到、长请求后到，制造“批次里有人结束、有人等待”的局面
	short := make([]*Request, 6)
	for i := range short {
		short[i] = eng.Submit([]int{1, 2, 3}, 4, 3, greedyParams(), uint64(10+i))
	}
	longPrompt := make([]int, 30)
	longR := eng.Submit(longPrompt, 30, 2, greedyParams(), 999)

	sawMixed := false
	finishCount := 0
	for !eng.Quiescent() {
		eng.Step()
		snap := eng.Snapshot()
		if len(snap.Decoding) > 0 && len(snap.Waiting) > 0 {
			sawMixed = true
		}
		finishCount = len(eng.Finished())
	}
	if !sawMixed {
		t.Fatal("从未出现“解码中 + 等待中”并存的连续批处理状态")
	}
	if finishCount != 7 {
		t.Fatalf("完成请求数错误: %d", finishCount)
	}
	if longR.State() != StateFinished {
		t.Fatal("长请求未完成")
	}
}
