package lm

import (
	"math"
	"testing"
)

func testCfg() Config {
	return Config{Vocab: 32, DModel: 16, Heads: 4, Layers: 2, DFF: 32, EOS: 31}
}

// 同种子两次初始化必须逐位相同。
func TestModelInitDeterministic(t *testing.T) {
	cfg := testCfg()
	a := NewModel(cfg, 123)
	b := NewModel(cfg, 123)
	if len(a.Emb) != len(b.Emb) {
		t.Fatal("embedding 行数不一致")
	}
	for i := range a.Emb {
		for j := range a.Emb[i] {
			if a.Emb[i][j] != b.Emb[i][j] {
				t.Fatalf("同种子初始化出现差异 (%d,%d)", i, j)
			}
		}
	}
}

// 不同种子必须产生不同权重。
func TestModelInitSeedDistinguishes(t *testing.T) {
	cfg := testCfg()
	a := NewModel(cfg, 1)
	b := NewModel(cfg, 2)
	diff := false
	for i := range a.Emb {
		for j := range a.Emb[i] {
			if a.Emb[i][j] != b.Emb[i][j] {
				diff = true
			}
		}
	}
	if !diff {
		t.Fatal("不同种子权重完全相同")
	}
}

// 带掩码的前向：被遮住的 key 对结果零影响。
// 对照实验：同一份已写入的 KV、同一个重算 query，唯一区别是掩码。
// 第二组把位置 2 的 K/V 污染并遮住，重算位置 1 的 logits 必须与干净组逐位相同。
func TestMaskIsolation(t *testing.T) {
	cfg := testCfg()

	run := func(pollute bool) []float64 {
		m := NewModel(cfg, 7)
		bm := NewBlockManager(cfg, 4, 8, 4)
		kv := bm.Acquire(1)
		bm.Grow(1)
		res := &Resolver{BM: bm, KV: kv}
		// 先写入 3 个位置的 KV
		m.Forward([]Segment{{
			Tokens: []int{1, 2, 3}, Positions: []int{0, 1, 2},
			ReadKV: res, WriteSink: res, Mask: causalMaskForTest(0, 3),
		}})
		kv.Length = 3
		if pollute {
			for l := 0; l < cfg.Layers; l++ {
				res.WriteK(l, 2, fillVec(cfg.DModel, 99.0))
				res.WriteV(l, 2, fillVec(cfg.DModel, -99.0))
			}
		}
		// 重算位置 1：可见位置 0、1；位置 2 一律遮住
		mask := [][]bool{{true, true, false}}
		out := m.Forward([]Segment{{
			Tokens: []int{2}, Positions: []int{1},
			ReadKV: res, Mask: mask,
		}})
		return out.Logits[0]
	}

	clean := run(false)
	masked := run(true)
	for v := range clean {
		if math.Abs(clean[v]-masked[v]) > 1e-12 {
			t.Fatalf("掩码隔离失败: %v vs %v", clean[v], masked[v])
		}
	}
}

func causalMaskForTest(oldLen, n int) [][]bool {
	mask := make([][]bool, n)
	for i := 0; i < n; i++ {
		row := make([]bool, oldLen+n)
		for j := 0; j < oldLen+i+1; j++ {
			row[j] = true
		}
		mask[i] = row
	}
	return mask
}

func fillVec(n int, v float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = v
	}
	return x
}

// 分页 KV 换出再换回：块表不变，数据逐位保留。
func TestSwapPreservesKV(t *testing.T) {
	cfg := testCfg()
	bm := NewBlockManager(cfg, 4, 4, 4)
	kv := bm.Acquire(1)
	bm.Grow(1)
	res := &Resolver{BM: bm, KV: kv}
	res.WriteK(0, 1, fillVec(cfg.DModel, 3.5))
	res.WriteV(1, 2, fillVec(cfg.DModel, -2.25))
	kv.Length = 3

	bm.SwapOut(1)
	if bm.GPUFree() != 4 {
		t.Fatalf("换出后 GPU 应全空, free=%d", bm.GPUFree())
	}
	bm.SwapIn(1)
	if got := res.K(0, 1)[0]; got != 3.5 {
		t.Fatalf("换回后 K 数据丢失: %v", got)
	}
	if got := res.V(1, 2)[0]; got != -2.25 {
		t.Fatalf("换回后 V 数据丢失: %v", got)
	}
	if kv.Length != 3 {
		t.Fatal("换页不应改变逻辑长度")
	}
}

// CompositeKV 拼接视图的下标分派。
func TestCompositeKV(t *testing.T) {
	cfg := testCfg()
	bm := NewBlockManager(cfg, 4, 4, 4)
	p := bm.Acquire(1)
	bm.Grow(1)
	pr := &Resolver{BM: bm, KV: p}
	pr.WriteK(0, 0, fillVec(cfg.DModel, 1.0))
	p.Length = 1
	s := NewSliceKV(cfg.Layers, cfg.DModel)
	s.WriteK(0, 0, fillVec(cfg.DModel, 2.0))
	c := &CompositeKV{Prefix: pr, Suffix: s}
	if c.Len() != 2 {
		t.Fatalf("拼接长度错误: %d", c.Len())
	}
	if c.K(0, 0)[0] != 1.0 || c.K(0, 1)[0] != 2.0 {
		t.Fatal("CompositeKV 下标分派错误")
	}
}

func TestPCGReproducible(t *testing.T) {
	a := NewPCG32(99, 3)
	b := NewPCG32(99, 3)
	for i := 0; i < 1000; i++ {
		if a.Uint32() != b.Uint32() {
			t.Fatal("PCG 同种子序列不一致")
		}
	}
	c := NewPCG32(99, 4)
	diff := false
	for i := 0; i < 1000; i++ {
		if a.Uint32() != c.Uint32() {
			diff = true
		}
	}
	if !diff {
		t.Fatal("不同 stream 序列竟然相同")
	}
}
