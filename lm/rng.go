package lm

import "math"

// PCG32 是 PCG-XSH-RR 随机数生成器，可跨平台复现。
// 全部采样（含固定种子模型初始化、采样请求）都走它，保证测试可逐位复现。
type PCG32 struct {
	state uint64
	inc   uint64
}

// NewPCG32 用 stream 选择不同的子序列，seed 决定起点。
func NewPCG32(seed, stream uint64) *PCG32 {
	r := &PCG32{inc: stream<<1 | 1}
	r.Uint32()
	r.state += seed
	r.Uint32()
	return r
}

func (r *PCG32) Uint32() uint32 {
	old := r.state
	r.state = old*6364136223846793005 + r.inc
	xorshifted := uint32(((old >> 18) ^ old) >> 27)
	rot := uint32(old >> 59)
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
}

// Float64 返回 [0,1) 的双精度浮点数，2^53 个等概率区间。
func (r *PCG32) Float64() float64 {
	// 32 位随机数取高 21 位作高位、再补 32 位，拼成 53 位尾数。
	hi := uint64(r.Uint32()>>11) << 32
	lo := uint64(r.Uint32())
	return float64(hi|lo) / (1 << 53)
}

// Norm 返回标准正态样本（Box-Muller），用于固定种子的模型初始化。
func (r *PCG32) Norm() float64 {
	u1 := r.Float64()
	if u1 < 1e-15 {
		u1 = 1e-15
	}
	u2 := r.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}
