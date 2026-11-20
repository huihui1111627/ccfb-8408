package engine

import (
	"embed"
	"encoding/json"
	"net/http"

	"lmsched/sample"
)

//go:embed web
var webFS embed.FS

// Handler 返回引擎监控页面的 HTTP handler。
// GET /           仪表盘页面
// GET /state      引擎状态 JSON（页面每秒轮询）
// POST /submit    手动追加一个请求（JSON body）
func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /state", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(e.Snapshot())
	})
	mux.HandleFunc("POST /submit", e.handleSubmit)
	// 静态仪表盘：/ 直接返回 index.html 内容；其它资源走嵌入文件系统。
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" || r.URL.Path == "" {
			data, err := webFS.ReadFile("web/index.html")
			if err != nil {
				http.Error(w, "dashboard not found", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = w.Write(data)
			return
		}
		http.FileServerFS(webFS).ServeHTTP(w, r)
	})
	return mux
}

type submitBody struct {
	PromptLen   int     `json:"prompt_len"`
	MaxNew      int     `json:"max_new"`
	Priority    int     `json:"priority"`
	Greedy      bool    `json:"greedy"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	RepPenalty  float64 `json:"rep_penalty"`
	Seed        uint64  `json:"seed"`
}

func (e *Engine) handleSubmit(w http.ResponseWriter, r *http.Request) {
	var b submitBody
	if err := json.NewDecoder(r.Body).Decode(&b); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if b.PromptLen <= 0 {
		b.PromptLen = 16
	}
	if b.MaxNew <= 0 {
		b.MaxNew = 32
	}
	if b.Priority == 0 {
		b.Priority = 3
	}
	if b.Seed == 0 {
		b.Seed = uint64(e.tick) + e.nextID*7919 + 1
	}
	prompt := make([]int, b.PromptLen)
	rng := newLocalRNG(b.Seed)
	for i := range prompt {
		prompt[i] = int(rng.Uint32() % uint32(e.target.Cfg.Vocab-1))
	}
	params := submitParams(b)
	req := e.Submit(prompt, b.MaxNew, b.Priority, params, b.Seed)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]uint64{"id": req.ID})
}

func submitParams(b submitBody) sample.Params {
	if b.Greedy {
		return sample.Params{Greedy: true, TopP: 1, RepetitionPenalty: 1}
	}
	temp := b.Temperature
	if temp <= 0 {
		temp = 1
	}
	topP := b.TopP
	if topP <= 0 || topP > 1 {
		topP = 1
	}
	rep := b.RepPenalty
	if rep <= 0 {
		rep = 1
	}
	return sample.Params{Temperature: temp, TopP: topP, RepetitionPenalty: rep}
}
