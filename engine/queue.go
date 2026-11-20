package engine

import "container/heap"

// queueItem 等待队列条目。换出待换回的请求永远排在新请求前面，
// 避免被反复抢占导致饥饿；其后按优先级、到达时间排序。
type queueItem struct {
	req *Request
	idx int
}

type priorityQueue []*queueItem

func (q priorityQueue) Len() int { return len(q) }

func (q priorityQueue) Less(i, j int) bool {
	a, b := q[i].req, q[j].req
	if a.swapBackWanted != b.swapBackWanted {
		return a.swapBackWanted
	}
	if a.Priority != b.Priority {
		return a.Priority > b.Priority
	}
	if a.arrivedTick != b.arrivedTick {
		return a.arrivedTick < b.arrivedTick
	}
	return a.ID < b.ID
}

func (q priorityQueue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
	q[i].idx = i
	q[j].idx = j
}

func (q *priorityQueue) Push(x any) {
	it := x.(*queueItem)
	it.idx = len(*q)
	*q = append(*q, it)
}

func (q *priorityQueue) Pop() any {
	old := *q
	n := len(old)
	it := old[n-1]
	old[n-1] = nil
	*q = old[:n-1]
	return it
}

type waitQueue struct {
	h priorityQueue
}

func newWaitQueue() *waitQueue {
	w := &waitQueue{}
	heap.Init(&w.h)
	return w
}

func (w *waitQueue) push(r *Request) {
	// 同一请求已在队列中时不重复入堆，避免一个 Request 对应多个 queueItem
	// 导致 remove 只删其一、残留条目让请求“既不在队列也不在 active”。
	for _, it := range w.h {
		if it.req == r {
			return
		}
	}
	heap.Push(&w.h, &queueItem{req: r})
}

func (w *waitQueue) pop() *Request {
	if w.h.Len() == 0 {
		return nil
	}
	return heap.Pop(&w.h).(*queueItem).req
}

func (w *waitQueue) peek() *Request {
	if w.h.Len() == 0 {
		return nil
	}
	return w.h[0].req
}

func (w *waitQueue) len() int { return w.h.Len() }

// remove 从队列中移除指定请求（用于挑队中可立即满足的换出请求优先换回）。
// 防御性地删除全部匹配项，保证移除后队列里不再残留该请求的条目。
func (w *waitQueue) remove(target *Request) {
	for i := 0; i < len(w.h); {
		if w.h[i].req == target {
			heap.Remove(&w.h, i)
			continue // 删除后下标平移，不递增
		}
		i++
	}
}

// ordered 快照（不弹出），供页面展示队列顺序。
func (w *waitQueue) ordered() []*Request {
	out := make([]*Request, 0, w.h.Len())
	for _, it := range w.h {
		out = append(out, it.req)
	}
	return out
}
