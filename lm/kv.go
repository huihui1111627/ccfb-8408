package lm

// Block 一个显存块：每层每槽位一份 K/V 向量。
// GPU 物理槽在系统生命周期内固定；换出时把 Block 的数据拷到 CPU 侧保存，
// 换回时拷到任意空闲 GPU 槽，因此物理槽可被不同请求在不同时间复用。
type Block struct {
	ID        int
	K, V      [][][]float64 // [layer][slot][dModel]
	blockSize int
}

func newBlock(id, layers, dim, blockSize int) *Block {
	b := &Block{ID: id, blockSize: blockSize}
	b.K = make([][][]float64, layers)
	b.V = make([][][]float64, layers)
	for l := range b.K {
		b.K[l] = make([][]float64, blockSize)
		b.V[l] = make([][]float64, blockSize)
		for s := 0; s < blockSize; s++ {
			b.K[l][s] = make([]float64, dim)
			b.V[l][s] = make([]float64, dim)
		}
	}
	return b
}

func (b *Block) bytes(dim int) int {
	return 2 * len(b.K) * b.blockSize * dim * 8
}

// cloneBlock 深拷贝一个块的数据（换出到 CPU 时用）。
func cloneBlock(src *Block) *Block {
	dst := &Block{ID: src.ID, blockSize: src.blockSize}
	dst.K = make([][][]float64, len(src.K))
	dst.V = make([][][]float64, len(src.V))
	for l := range src.K {
		dst.K[l] = make([][]float64, len(src.K[l]))
		dst.V[l] = make([][]float64, len(src.V[l]))
		for s := range src.K[l] {
			dst.K[l][s] = append([]float64(nil), src.K[l][s]...)
			dst.V[l][s] = append([]float64(nil), src.V[l][s]...)
		}
	}
	return dst
}

// copyBlockData 把 src 的 K/V 数据拷进已存在的同形状 dst。
func copyBlockData(dst, src *Block) {
	for l := range src.K {
		for s := range src.K[l] {
			copy(dst.K[l][s], src.K[l][s])
			copy(dst.V[l][s], src.V[l][s])
		}
	}
}

// PagedKV 一个请求的分页 KV：逻辑块经 Table 映射到 GPU 物理槽。
// 换出期间 Table 不变（逻辑结构保留），数据在 BlockManager 的 cpuStore 里
// 按“请求 ID + 逻辑块号”暂存，与物理槽 ID 解耦，避免槽复用时键冲突；
// 换回时 Table 被改写为新分配的 GPU 槽，但逻辑位置与内容完全不变。
type PagedKV struct {
	RequestID uint64
	Table     []int // 逻辑块号 -> GPU 物理槽 ID（换出期间指向的槽已被回收）
	Length    int   // 已提交的 token 数（含预填充）
	blockSize int
}

// cpuKey 换出块在 cpuStore 里的唯一键：请求 ID + 逻辑块号。
type cpuKey struct {
	reqID  uint64
	logBlk int
}

// BlockManager 管理固定的 GPU 物理槽池与 CPU 换出存储。
type BlockManager struct {
	cfg       Config
	blockSize int

	gpuSlots map[int]*Block    // 物理槽 ID -> 槽对象（固定存在）
	gpuFree  []int             // 当前空闲的 GPU 物理槽 ID
	owned    map[int]bool      // 槽 ID -> 是否正被某请求占用（未在空闲栈）
	cpuStore map[cpuKey]*Block // 换出块暂存（按请求+逻辑块索引）

	allocs map[uint64]*PagedKV

	SwapBytes int64
	SwapsOut  int
	SwapsIn   int
}

func NewBlockManager(cfg Config, blockSize, gpuCount, cpuCount int) *BlockManager {
	bm := &BlockManager{
		cfg:       cfg,
		blockSize: blockSize,
		gpuSlots:  make(map[int]*Block),
		cpuStore:  make(map[cpuKey]*Block),
		owned:     make(map[int]bool),
		allocs:    make(map[uint64]*PagedKV),
	}
	for i := 0; i < gpuCount; i++ {
		bm.gpuSlots[i] = newBlock(i, cfg.Layers, cfg.DModel, blockSize)
		bm.gpuFree = append(bm.gpuFree, i)
	}
	_ = cpuCount // CPU 侧按块动态分配，无需预开固定槽
	return bm
}

func (bm *BlockManager) BlockSize() int { return bm.blockSize }
func (bm *BlockManager) GPUFree() int   { return len(bm.gpuFree) }
func (bm *BlockManager) CPUFree() int   { return 0 }
func (bm *BlockManager) GPUTotal() int  { return len(bm.gpuSlots) }
func (bm *BlockManager) CPUTotal() int  { return len(bm.cpuStore) }

// GPUBlockIDs 所有 GPU 物理槽 ID（含占用与空闲）。
func (bm *BlockManager) GPUBlockIDs() []int {
	out := make([]int, 0, len(bm.gpuSlots))
	for id := range bm.gpuSlots {
		out = append(out, id)
	}
	return out
}

// CPUBlockIDs 当前存放在 CPU 的块对应的【源 GPU 槽 ID】（快照展示用）。
func (bm *BlockManager) CPUBlockIDs() []int {
	out := make([]int, 0, len(bm.cpuStore))
	seen := make(map[int]bool)
	for k := range bm.cpuStore {
		// 用暂存块对象记录的源槽 ID（Block.ID）
		if id := bm.cpuStore[k].ID; !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	return out
}

// CPUOwners 快照：每个被换出到 CPU 的块（源槽 ID）所属的请求 ID。
func (bm *BlockManager) CPUOwners() map[int]uint64 {
	out := make(map[int]uint64, len(bm.cpuStore))
	for k, blk := range bm.cpuStore {
		out[blk.ID] = k.reqID
	}
	return out
}

// Allocs 快照：请求 ID -> 其物理块表。
func (bm *BlockManager) Allocs() map[uint64][]int {
	out := make(map[uint64][]int, len(bm.allocs))
	for id, kv := range bm.allocs {
		out[id] = append([]int(nil), kv.Table...)
	}
	return out
}

// ReadK / ReadV 按 GPU 物理槽 ID 读取向量，供测试与快照使用。
func (bm *BlockManager) ReadK(phys, layer, slot int) []float64 {
	return bm.gpuSlots[phys].K[layer][slot]
}

func (bm *BlockManager) ReadV(phys, layer, slot int) []float64 {
	return bm.gpuSlots[phys].V[layer][slot]
}

// Acquire 为请求建立空 KV 句柄。
func (bm *BlockManager) Acquire(reqID uint64) *PagedKV {
	kv := &PagedKV{RequestID: reqID, blockSize: bm.blockSize}
	bm.allocs[reqID] = kv
	return kv
}

func (bm *BlockManager) Get(reqID uint64) *PagedKV { return bm.allocs[reqID] }

// Grow 给请求追加一个 GPU 槽；无空闲槽时返回 false（调度器据此触发换出）。
func (bm *BlockManager) Grow(reqID uint64) (int, bool) {
	kv := bm.allocs[reqID]
	if len(bm.gpuFree) == 0 {
		return 0, false
	}
	id := bm.gpuFree[len(bm.gpuFree)-1]
	bm.gpuFree = bm.gpuFree[:len(bm.gpuFree)-1]
	bm.owned[id] = true
	kv.Table = append(kv.Table, id)
	return id, true
}

// Free 归还请求占用的全部 GPU 槽，并清理其 CPU 暂存（请求结束）。
func (bm *BlockManager) Free(reqID uint64) {
	kv := bm.allocs[reqID]
	if kv == nil {
		return
	}
	for lb, id := range kv.Table {
		if bm.owned[id] {
			bm.owned[id] = false
			bm.gpuFree = append(bm.gpuFree, id)
		}
		delete(bm.cpuStore, cpuKey{reqID, lb})
	}
	delete(bm.allocs, reqID)
}

// SwapOut 把请求的全部块数据拷到 CPU 暂存，并释放它们占用的 GPU 槽。
// 按“请求+逻辑块”存，与物理槽 ID 解耦；逻辑长度不变，无需重算前缀。
func (bm *BlockManager) SwapOut(reqID uint64) int {
	kv := bm.allocs[reqID]
	moved := 0
	for lb, id := range kv.Table {
		if !bm.owned[id] {
			continue // 该槽当前不被本请求占用（已在空闲栈），跳过
		}
		saved := cloneBlock(bm.gpuSlots[id])
		saved.ID = id // 记录源槽，仅用于展示
		bm.cpuStore[cpuKey{reqID, lb}] = saved
		bm.owned[id] = false
		bm.gpuFree = append(bm.gpuFree, id)
		bm.SwapBytes += int64(bm.gpuSlots[id].bytes(bm.cfg.DModel))
		moved++
	}
	bm.SwapsOut++
	return moved
}

// CanSwapIn 是否有足够 GPU 空闲槽把请求换回。
func (bm *BlockManager) CanSwapIn(reqID uint64) bool {
	kv := bm.allocs[reqID]
	need := 0
	for lb := range kv.Table {
		if _, onCPU := bm.cpuStore[cpuKey{reqID, lb}]; onCPU {
			need++
		}
	}
	return len(bm.gpuFree) >= need
}

// SwapIn 把请求的 CPU 暂存块搬到任意空闲 GPU 槽。
// 重写块表为新物理槽 ID；逻辑位置与 K/V 内容保持不变。
// 返回实际迁移的块数；仅当确有块迁回时才累计 SwapsIn 计数。
func (bm *BlockManager) SwapIn(reqID uint64) int {
	kv := bm.allocs[reqID]
	moved := 0
	for lb := range kv.Table {
		key := cpuKey{reqID, lb}
		saved, onCPU := bm.cpuStore[key]
		if !onCPU {
			continue
		}
		if len(bm.gpuFree) == 0 {
			break // 空闲槽耗尽，已迁回的保留，剩余下轮继续
		}
		newPhys := bm.gpuFree[len(bm.gpuFree)-1]
		bm.gpuFree = bm.gpuFree[:len(bm.gpuFree)-1]
		copyBlockData(bm.gpuSlots[newPhys], saved)
		bm.owned[newPhys] = true
		delete(bm.cpuStore, key)
		kv.Table[lb] = newPhys
		bm.SwapBytes += int64(saved.bytes(bm.cfg.DModel))
		moved++
	}
	if moved > 0 {
		bm.SwapsIn++
	}
	return moved
}

func (p *PagedKV) Len() int { return p.Length }

func (bm *BlockManager) blockOf(p *PagedKV, j int) (*Block, int) {
	logical := j / p.blockSize
	phys := p.Table[logical]
	slot := j % p.blockSize
	return bm.gpuSlots[phys], slot
}

// Resolver 把块管理器与请求句柄绑定成可前向的 KVResolver。
type Resolver struct {
	BM *BlockManager
	KV *PagedKV
}

func (r *Resolver) Len() int { return r.KV.Length }
func (r *Resolver) K(layer, j int) []float64 {
	blk, slot := r.BM.blockOf(r.KV, j)
	return blk.K[layer][slot]
}
func (r *Resolver) V(layer, j int) []float64 {
	blk, slot := r.BM.blockOf(r.KV, j)
	return blk.V[layer][slot]
}
func (r *Resolver) WriteK(layer, j int, vec []float64) {
	blk, slot := r.BM.blockOf(r.KV, j)
	copy(blk.K[layer][slot], vec)
}
func (r *Resolver) WriteV(layer, j int, vec []float64) {
	blk, slot := r.BM.blockOf(r.KV, j)
	copy(blk.V[layer][slot], vec)
}

// WriteBase 分页视图：槽位预分配，新项从当前逻辑长度开始写；不修改 Length，
// 逻辑长度由引擎在 Forward 返回后统一提交。
func (r *Resolver) WriteBase(n int) int { return r.KV.Length }

// SliceKV 连续的瞬态 KV（草稿序列、树分支），可自由追加。
type SliceKV struct {
	Layers int
	Dim    int
	Ks, Vs [][][]float64 // [layer][pos][dim]
}

func NewSliceKV(layers, dim int) *SliceKV {
	s := &SliceKV{Layers: layers, Dim: dim}
	s.Ks = make([][][]float64, layers)
	s.Vs = make([][][]float64, layers)
	return s
}

func (s *SliceKV) Len() int {
	if len(s.Ks[0]) == 0 {
		return 0
	}
	return len(s.Ks[0])
}

func (s *SliceKV) ensure(j int) {
	for len(s.Ks[0]) <= j {
		for l := 0; l < s.Layers; l++ {
			s.Ks[l] = append(s.Ks[l], make([]float64, s.Dim))
			s.Vs[l] = append(s.Vs[l], make([]float64, s.Dim))
		}
	}
}

func (s *SliceKV) WriteK(layer, j int, vec []float64) {
	s.ensure(j)
	copy(s.Ks[layer][j], vec)
}
func (s *SliceKV) WriteV(layer, j int, vec []float64) {
	s.ensure(j)
	copy(s.Vs[layer][j], vec)
}
func (s *SliceKV) K(layer, j int) []float64 { return s.Ks[layer][j] }
func (s *SliceKV) V(layer, j int) []float64 { return s.Vs[layer][j] }

// WriteBase 瞬态视图：扩容到 old+n，返回旧长度作为写入起点。
func (s *SliceKV) WriteBase(n int) int {
	base := s.Len()
	for j := base; j < base+n; j++ {
		s.ensure(j)
	}
	return base
}

// Trim 截断到前 n 个位置；n 不小于当前长度时为空操作。
func (s *SliceKV) Trim(n int) {
	cur := len(s.Ks[0])
	if n >= cur {
		return
	}
	for l := 0; l < s.Layers; l++ {
		s.Ks[l] = s.Ks[l][:n]
		s.Vs[l] = s.Vs[l][:n]
	}
}

// CompositeKV 前缀（分页）+ 后缀（连续瞬态）拼接视图，树/链验证用。
type CompositeKV struct {
	Prefix KVResolver
	Suffix KVResolver
}

func (c *CompositeKV) Len() int { return c.Prefix.Len() + c.Suffix.Len() }
func (c *CompositeKV) K(layer, j int) []float64 {
	if j < c.Prefix.Len() {
		return c.Prefix.K(layer, j)
	}
	return c.Suffix.K(layer, j-c.Prefix.Len())
}
func (c *CompositeKV) V(layer, j int) []float64 {
	if j < c.Prefix.Len() {
		return c.Prefix.V(layer, j)
	}
	return c.Suffix.V(layer, j-c.Prefix.Len())
}
func (c *CompositeKV) WriteK(layer, j int, vec []float64) {
	if j < c.Prefix.Len() {
		c.Prefix.WriteK(layer, j, vec)
		return
	}
	c.Suffix.WriteK(layer, j-c.Prefix.Len(), vec)
}
func (c *CompositeKV) WriteV(layer, j int, vec []float64) {
	if j < c.Prefix.Len() {
		c.Prefix.WriteV(layer, j, vec)
		return
	}
	c.Suffix.WriteV(layer, j-c.Prefix.Len(), vec)
}

// WriteBase 拼接视图：写入永远落在后缀上，全局下标 = 前缀长度 + 后缀写入起点。
func (c *CompositeKV) WriteBase(n int) int {
	return c.Prefix.Len() + c.Suffix.WriteBase(n)
}
