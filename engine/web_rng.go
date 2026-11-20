package engine

import "lmsched/lm"

// newLocalRNG 给 web 手动提交生成提示词用，独立于请求自身的采样流。
func newLocalRNG(seed uint64) *lm.PCG32 {
	return lm.NewPCG32(seed, 5555)
}
