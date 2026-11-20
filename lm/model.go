package lm

import (
	"math"
)

// Config 描述一个微型因果 Transformer。模型本身只是调度的载体，
// 因此刻意保持很小，保证纯 Go 也能在测试里跑几十万次前向。
type Config struct {
	Vocab  int
	DModel int
	Heads  int
	Layers int
	DFF    int
	EOS    int
}

func (c Config) HeadDim() int { return c.DModel / c.Heads }

// Layer 单层参数。矩阵统一行主序 [out][in]。
type Layer struct {
	WQ [][]float64
	WK [][]float64
	WV [][]float64
	WO [][]float64
	W1 [][]float64
	B1 []float64
	W2 [][]float64
}

// Model 固定种子随机初始化的小语言模型，权重只取决于 seed。
type Model struct {
	Cfg    Config
	Emb    [][]float64 // [V][D]
	Layers []Layer
	Head   [][]float64 // 输出投影 [V][D]
}

// NewModel 用固定种子初始化。每个张量拿独立的子流，
// 调整结构或初始化顺序不会让其它张量的随机数序列错位。
func NewModel(cfg Config, seed uint64) *Model {
	d := cfg.DModel
	embStd := 1.0 / math.Sqrt(float64(d))
	m := &Model{Cfg: cfg}
	rngEmb := NewPCG32(seed, 1)
	m.Emb = randMatrix(cfg.Vocab, d, embStd, rngEmb)

	matStd := 1.0 / math.Sqrt(float64(d))
	for l := 0; l < cfg.Layers; l++ {
		streamBase := uint64(10 + l*8)
		layer := Layer{
			WQ: randMatrix(d, d, matStd, NewPCG32(seed, streamBase+0)),
			WK: randMatrix(d, d, matStd, NewPCG32(seed, streamBase+1)),
			WV: randMatrix(d, d, matStd, NewPCG32(seed, streamBase+2)),
			WO: randMatrix(d, d, matStd, NewPCG32(seed, streamBase+3)),
			W1: randMatrix(cfg.DFF, d, matStd, NewPCG32(seed, streamBase+4)),
			B1: make([]float64, cfg.DFF),
			W2: randMatrix(d, cfg.DFF, matStd, NewPCG32(seed, streamBase+5)),
		}
		m.Layers = append(m.Layers, layer)
	}
	m.Head = randMatrix(cfg.Vocab, d, embStd, NewPCG32(seed, 99))
	return m
}

func randMatrix(rows, cols int, std float64, rng *PCG32) [][]float64 {
	a := make([][]float64, rows)
	for i := range a {
		row := make([]float64, cols)
		for j := range row {
			row[j] = rng.Norm() * std
		}
		a[i] = row
	}
	return a
}

// KVResolver 把“逻辑 token 位置”映射到分页 KV 存储。
// 普通解码由块管理器实现；草稿/树的瞬态序列用可拼接的 SliceKV 实现。
type KVResolver interface {
	Len() int
	K(layer, j int) []float64
	V(layer, j int) []float64
	WriteK(layer, j int, vec []float64)
	WriteV(layer, j int, vec []float64)
	// WriteBase 声明一次写入 n 个新项时，第一项在视图里的下标。
	WriteBase(n int) int
}

// Segment 一次前向里的一段：连续的 token/位置，配读视图、写槽和掩码。
type Segment struct {
	Tokens    []int
	Positions []int
	// ReadKV 注意力读取的 KV 视图（含全部历史，掩码按下标访问）。
	ReadKV KVResolver
	// WriteSink 本段新 K/V 的写入槽；nil 表示不写回。
	// 它与 ReadKV 可以不同对象：验证时读“分页前缀+目标瞬态”，写只追加到目标瞬态。
	WriteSink KVResolver
	// Mask[i][j]：本段第 i 个 query 是否能看到 ReadKV 里的第 j 项。
	// nil 表示纯因果（仅当 ReadKV 恰好按本段 n 项排列时合法）。
	Mask [][]bool
}

// writeEnabled 是否写回 K/V。
func (s *Segment) writeEnabled() bool { return s.WriteSink != nil }

// kv 读取视图的兼容访问。
func (s *Segment) kv() KVResolver { return s.ReadKV }

// FwdStats 本次前向的工作量统计，用于吞吐/虚拟时间核算。
type FwdStats struct {
	Queries      int
	AttnPairs    int // 所有层的 (query, 可见 key) 对数
	MatmulMACs   int // 除注意力外的矩阵乘累加数
	ForwardUnits int // 综合 FLOPs 代理值
}

// Output 前向结果：Logits 按 Segment 顺序、段内按位置排列。
type Output struct {
	Logits [][]float64
	Stats  FwdStats
}

const lnEps = 1e-5

// Forward 对若干段做一次前向。同一次调用里的段彼此通过掩码隔离
// （树验证时各分支互不可见），但共享一次调用的统计口径。
func (m *Model) Forward(segs []Segment) *Output {
	cfg := m.Cfg
	hd := cfg.HeadDim()
	out := &Output{}

	for si, seg := range segs {
		n := len(seg.Tokens)
		if n == 0 {
			continue // 空段不产生输出，避免空掩码越界
		}
		// 1) token 嵌入 + 正弦位置编码
		h := make([][]float64, n)
		for i := 0; i < n; i++ {
			h[i] = make([]float64, cfg.DModel)
			copy(h[i], m.Emb[seg.Tokens[i]])
			addPositional(h[i], seg.Positions[i])
		}

		readView := seg.ReadKV

		// 写槽先声明本段 n 项的写入起点（相对写槽自身下标）并完成必要扩容。
		writeBase := 0
		if seg.writeEnabled() {
			writeBase = seg.WriteSink.WriteBase(n)
		}

		// 注意力遍历长度以【掩码行数】为准；nil 掩码（纯段、读写同一视图）
		// 时视图长度恰为旧 Len+n。这样独立写槽经 composite 后缀扩容后的
		// 读视图长度、与调用方预算的掩码长度严格对齐。
		var kvLen int
		if seg.Mask != nil {
			kvLen = len(seg.Mask[0])
		} else {
			kvLen = readView.Len()
		}

		for l := 0; l < cfg.Layers; l++ {
			layer := &m.Layers[l]
			residual := cloneRows(h)

			// 2) Q/K/V 投影
			qs := make([][]float64, n)
			ks := make([][]float64, n)
			vs := make([][]float64, n)
			for i := 0; i < n; i++ {
				qs[i] = matVec(layer.WQ, h[i])
				ks[i] = matVec(layer.WK, h[i])
				vs[i] = matVec(layer.WV, h[i])
				out.Stats.MatmulMACs += 3 * cfg.DModel * cfg.DModel
				if seg.writeEnabled() {
					seg.WriteSink.WriteK(l, writeBase+i, ks[i])
					seg.WriteSink.WriteV(l, writeBase+i, vs[i])
				}
			}

			// 3) 掩码多头注意力：每个头独立打分/softmax，再按维度拼回
			attnOut := make([][]float64, n)
			for i := 0; i < n; i++ {
				merged := make([]float64, cfg.DModel)
				for head := 0; head < cfg.Heads; head++ {
					lo, hi := head*hd, (head+1)*hd
					scores := make([]float64, kvLen)
					visible := 0
					for j := 0; j < kvLen; j++ {
						// nil 掩码 = 纯因果段：读视图恰好按“本段 n 项”排列，
						// query i 只能看到 0..i。带历史的前向必须显式传掩码。
						vis := seg.Mask == nil && j <= i
						if seg.Mask != nil {
							vis = seg.Mask[i][j]
						}
						if !vis {
							scores[j] = math.Inf(-1)
							continue
						}
						visible++
						s := 0.0
						kj := readView.K(l, j)
						for d := lo; d < hi; d++ {
							s += qs[i][d] * kj[d]
						}
						scores[j] = s / math.Sqrt(float64(hd))
					}
					probs := softmax(scores)
					out.Stats.AttnPairs += visible
					for j := 0; j < kvLen; j++ {
						if probs[j] == 0 {
							continue
						}
						vj := readView.V(l, j)
						for d := lo; d < hi; d++ {
							merged[d] += probs[j] * vj[d]
						}
					}
				}
				attnOut[i] = matVec(layer.WO, merged)
				out.Stats.MatmulMACs += cfg.DModel * cfg.DModel
			}

			// 4) 残差 + LayerNorm
			for i := 0; i < n; i++ {
				for d := 0; d < cfg.DModel; d++ {
					h[i][d] = residual[i][d] + attnOut[i][d]
				}
				layerNorm(h[i])
			}

			// 5) FFN（GELU 精确版）+ 残差 + LayerNorm
			for i := 0; i < n; i++ {
				pre := make([]float64, cfg.DFF)
				for o := 0; o < cfg.DFF; o++ {
					pre[o] = layer.B1[o] + dot(layer.W1[o], h[i])
					pre[o] = gelu(pre[o])
				}
				ffnOut := make([]float64, cfg.DModel)
				for d := 0; d < cfg.DModel; d++ {
					s := 0.0
					for o := 0; o < cfg.DFF; o++ {
						s += layer.W2[d][o] * pre[o]
					}
					ffnOut[d] = h[i][d] + s
				}
				layerNorm(ffnOut)
				h[i] = ffnOut
				out.Stats.MatmulMACs += 2 * cfg.DModel * cfg.DFF
			}
		}

		// 6) 输出投影。默认返回段内全部位置的 logits，
		// 预填充只取最后一个，树验证每个节点都要。
		for i := 0; i < n; i++ {
			logits := make([]float64, cfg.Vocab)
			for v := 0; v < cfg.Vocab; v++ {
				logits[v] = dot(m.Head[v], h[i])
			}
			out.Logits = append(out.Logits, logits)
			out.Stats.MatmulMACs += cfg.DModel * cfg.Vocab
		}
		out.Stats.Queries += n
		_ = si
	}

	// ForwardUnits：注意力按头内点积 4*hd 每对，其余按 2*MAC 折算。
	out.Stats.ForwardUnits = out.Stats.AttnPairs*4*cfg.HeadDim()*cfg.Heads +
		out.Stats.MatmulMACs*2
	return out
}

func addPositional(x []float64, pos int) {
	d := len(x)
	for i := 0; i < d; i += 2 {
		denom := math.Pow(10000, float64(i)/float64(d))
		x[i] += math.Sin(float64(pos) / denom)
		if i+1 < d {
			x[i+1] += math.Cos(float64(pos) / denom)
		}
	}
}

func matVec(a [][]float64, x []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = dot(a[i], x)
	}
	return out
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func softmax(x []float64) []float64 {
	max := math.Inf(-1)
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		if math.IsInf(v, -1) {
			out[i] = 0
			continue
		}
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	if sum == 0 {
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func layerNorm(x []float64) {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	var sq float64
	for _, v := range x {
		d := v - mean
		sq += d * d
	}
	std := math.Sqrt(sq/float64(len(x)) + lnEps)
	for i := range x {
		x[i] = (x[i] - mean) / std
	}
}

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
}

func cloneRows(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[i]))
		copy(out[i], a[i])
	}
	return out
}
