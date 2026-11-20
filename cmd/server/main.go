package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"lmsched/engine"
	"lmsched/lm"
	"lmsched/sample"
)

func main() {
	addr := flag.String("addr", ":8080", "HTTP 监听地址")
	seed := flag.Uint64("seed", 20261121, "模型固定随机种子")
	vocab := flag.Int("vocab", 64, "词表大小")
	dim := flag.Int("dim", 48, "模型隐藏维")
	heads := flag.Int("heads", 4, "注意力头数")
	layers := flag.Int("layers", 2, "层数")
	dff := flag.Int("dff", 96, "FFN 中间维")
	draftScale := flag.Float64("draft-scale", 0.5, "草稿模型相对目标模型的隐藏维比例")

	blockSize := flag.Int("block-size", 8, "显存块大小（token）")
	gpuBlocks := flag.Int("gpu-blocks", 48, "GPU 块数")
	cpuBlocks := flag.Int("cpu-blocks", 64, "CPU 块数")
	maxBatch := flag.Int("max-batch", 16, "最大并发请求数")
	chunk := flag.Int("prefill-chunk", 8, "预填充分段 token 数")
	spec := flag.String("spec", "chain", "投机解码模式: none|chain|tree")
	kmin := flag.Int("kmin", 1, "自适应 k 下限")
	kmax := flag.Int("kmax", 6, "自适应 k 上限")
	branches := flag.Int("branches", 3, "树模式分支数")
	tickMs := flag.Int("tick-ms", 100, "每个调度 tick 的墙钟间隔")

	autoload := flag.Int("autoload", 0, "启动时自动生成的混合请求数（0 不生成）")
	flag.Parse()

	targetCfg := lm.Config{
		Vocab: *vocab, DModel: *dim, Heads: *heads, Layers: *layers,
		DFF: *dff, EOS: *vocab - 1,
	}
	draftDim := *dim / 2
	if draftDim < 8 {
		draftDim = 8
	}
	if draftDim%*heads != 0 {
		draftDim = (*heads) * (draftDim / *heads)
		if draftDim == 0 {
			draftDim = *heads
		}
	}
	draftCfg := lm.Config{
		Vocab: *vocab, DModel: draftDim, Heads: *heads, Layers: *layers,
		DFF: draftDim * 2, EOS: *vocab - 1,
	}
	_ = draftScale

	target := lm.NewModel(targetCfg, *seed)
	draft := lm.NewModel(draftCfg, *seed+1)

	mode := engine.SpecNone
	switch *spec {
	case "chain":
		mode = engine.SpecChain
	case "tree":
		mode = engine.SpecTree
	}
	eng := engine.NewEngine(target, draft, engine.Config{
		PrefillChunkSize: *chunk,
		Spec:             mode,
		KMin:             *kmin,
		KMax:             *kmax,
		TreeBranches:     *branches,
		BlockSize:        *blockSize,
		GPUBlocks:        *gpuBlocks,
		CPUBlocks:        *cpuBlocks,
		MaxBatch:         *maxBatch,
	})

	if *autoload > 0 {
		wl := engine.GenerateWorkload(targetCfg, engine.WorkloadSpec{
			Requests:    *autoload,
			Seed:        *seed + 77,
			ShortPrompt: [2]int{4, 16},
			LongPrompt:  [2]int{48, 96},
			ShortOut:    [2]int{8, 24},
			LongOut:     [2]int{24, 64},
			LongRatio:   0.25,
			GreedyRatio: 0.4,
		})
		for _, w := range wl {
			eng.Submit(w.Prompt, w.MaxNew, w.Priority, w.Params, w.Seed)
		}
		log.Printf("已自动生成 %d 个长短混杂请求", *autoload)
	}

	go func() {
		ticker := time.NewTicker(time.Duration(*tickMs) * time.Millisecond)
		defer ticker.Stop()
		for range ticker.C {
			eng.Step()
		}
	}()

	log.Printf("调度核心启动: 目标 d=%d 层=%d 草稿 d=%d, 模式=%s, 页面 http://localhost%s",
		*dim, *layers, draftDim, *spec, *addr)
	srv := &http.Server{Addr: *addr, Handler: eng.Handler()}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
	_ = sample.Params{}
	fmt.Println()
}
