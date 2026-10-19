# -*- coding: utf-8 -*-
"""中文文献搜索引擎 - 单文件演示脚本(界面 + 服务一体)。

运行后自动打开浏览器, 在网页中检索约 29 万篇文献:
    python webapp/search.py                       # BM25 关键词召回(纯 CPU, 默认)
    python webapp/search.py --engine ernie        # ERNIE 向量召回(需 paddle + index/)
    python webapp/search.py --engine rerank       # BM25 召回 + ERNIE-3.0 精排(需 paddle)

三个引擎:
  bm25    纯 CPU, 任何机器可跑(默认)
  ernie   ERNIE 双塔 + HNSW 向量召回, 需 paddle 与离线构建的 index/
  rerank  BM25 召回 + 交叉编码器精排, 额外需要 weights/rank/

依赖: bm25 引擎仅需 jieba+numpy; ernie/rerank 另需 paddlepaddle+paddlenlp。
"""
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bm25 import BM25Index, tokenize
from src.utils import load_corpus


# ================================================================ 内嵌网页
EMBEDDED_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>中文文献语义检索引擎</title>
<style>
  :root{
    --blue:#1a56db; --blue-dark:#1641a3; --ink:#1f2937; --ink2:#6b7280;
    --line:#e5e7eb; --mark:#fff3a3;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{font-family:"Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;background:#fff;color:var(--ink);}
  a{color:var(--blue);text-decoration:none;}

  .bar{display:flex;align-items:center;gap:14px;padding:14px 24px;border-bottom:1px solid var(--line);
       position:sticky;top:0;background:#fff;z-index:10;}
  .logo{font-size:19px;font-weight:700;letter-spacing:1px;cursor:pointer;white-space:nowrap;}
  .logo b{color:var(--blue);}
  .searchbox{flex:1;display:flex;max-width:640px;}
  .searchbox input{flex:1;border:2px solid var(--blue);border-right:none;border-radius:8px 0 0 8px;
       padding:9px 14px;font-size:15px;outline:none;}
  .searchbox button{border:none;background:var(--blue);color:#fff;padding:0 22px;font-size:15px;
       border-radius:0 8px 8px 0;cursor:pointer;}
  .searchbox button:hover{background:var(--blue-dark);}
  select{border:1px solid var(--line);border-radius:6px;padding:7px 8px;font-size:13px;color:var(--ink2);
       background:#fff;cursor:pointer;}

  .hero{max-width:680px;margin:12vh auto 0;text-align:center;padding:0 20px;}
  .hero h1{font-size:34px;font-weight:700;margin-bottom:10px;}
  .hero h1 b{color:var(--blue);}
  .hero p{color:var(--ink2);font-size:14px;margin-bottom:28px;}
  .chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;}
  .chip{border:1px solid var(--line);border-radius:999px;padding:7px 16px;font-size:13.5px;
       color:var(--ink2);cursor:pointer;background:#fafafa;}
  .chip:hover{border-color:var(--blue);color:var(--blue);background:#f0f6ff;}

  .wrap{max-width:820px;margin:0 auto;padding:18px 20px 60px;}
  .meta{color:var(--ink2);font-size:12.5px;margin:14px 0 6px;}
  .meta b{color:var(--ink);}
  .result{padding:14px 4px;border-bottom:1px solid #f1f3f5;}
  .result .head{display:flex;align-items:center;gap:10px;}
  .result .rank{color:#c3c9d4;font-size:12px;font-weight:700;width:20px;flex:none;}
  .result .score{color:var(--ink2);font-size:12px;white-space:nowrap;}
  .result .scorebar{flex:1;height:4px;background:#f1f3f5;border-radius:2px;overflow:hidden;}
  .result .scorebar i{display:block;height:100%;background:linear-gradient(90deg,#7fb1ff,var(--blue));border-radius:2px;}
  .result .title{font-size:15.5px;line-height:1.55;margin:6px 0 4px;}
  .result .docid{font-size:12px;color:#9ca3af;}
  mark{background:var(--mark);padding:0 1px;border-radius:2px;}
  .empty{color:var(--ink2);text-align:center;padding:60px 0;font-size:14px;}
  .err{color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px 16px;
       font-size:13.5px;margin-top:16px;}

  .foot{margin-top:40px;color:#9ca3af;font-size:12px;text-align:center;line-height:1.8;}
  @media (max-width:640px){ .bar{flex-wrap:wrap;} .searchbox{max-width:none;width:100%;} }
</style>
</head>
<body>

<div class="bar">
  <div class="logo" onclick="resetHome()"><b>文献</b>检索引擎</div>
  <div class="searchbox">
    <input id="q" type="text" placeholder="输入文献标题或关键词，回车检索…" autofocus>
    <button onclick="doSearch()">检索</button>
  </div>
  <select id="topk" title="返回条数">
    <option value="10" selected>前 10 条</option>
    <option value="20">前 20 条</option>
    <option value="50">前 50 条</option>
  </select>
</div>

<div class="hero" id="hero">
  <h1>中文<b>文献</b>语义检索引擎</h1>
  <p>约 29 万篇文献 · 双路召回 + 精排 · 机器学习课程设计</p>
  <div class="chips" id="chips"></div>
</div>

<div class="wrap" id="output" style="display:none"></div>
<div class="foot" id="foot"></div>

<script>
const SAMPLES = [
  "热处理对尼龙6结晶行为的影响",
  "冠状动脉粥样硬化诊断",
  "W6Mo5Cr4V2钢淬火开裂",
  "跨境电子商务发展策略研究",
  "大学英语委婉语翻译策略",
];

const $ = id => document.getElementById(id);

fetch("/api/health").then(r => r.json()).then(h => {
  $("foot").innerHTML =
    "引擎：<b>" + h.label + "</b> ｜ 语料 " + h.docs.toLocaleString() +
    " 篇（加载 " + h.load_seconds + "s）｜ <a href='https://github.com/fuyi-literature-search/literature-search' target='_blank'>GitHub 仓库</a>";
}).catch(() => {
  $("foot").textContent = "无法连接搜索服务, 请确认 search.py 正在运行";
});
SAMPLES.forEach(s => {
  const c = document.createElement("div");
  c.className = "chip"; c.textContent = s;
  c.onclick = () => { $("q").value = s; doSearch(); };
  $("chips").appendChild(c);
});

$("q").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function highlight(text, tokens) {
  let html = escapeHtml(text);
  // 按 token 长度降序替换, 避免短词先匹配破坏长词
  (tokens || []).filter(t => t.length >= 2).sort((a, b) => b.length - a.length).forEach(t => {
    const safe = t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    html = html.replace(new RegExp(safe, "gi"), m => "<mark>" + m + "</mark>");
  });
  return html;
}

async function doSearch() {
  const q = $("q").value.trim();
  if (!q) return;
  $("hero").style.display = "none";
  $("output").style.display = "block";
  $("output").innerHTML = "<div class='empty'>检索中…</div>";

  const url = "/api/search?q=" + encodeURIComponent(q) + "&topk=" + $("topk").value;
  try {
    const r = await fetch(url);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
    render(data);
  } catch (e) {
    $("output").innerHTML = "<div class='err'>检索失败：" + escapeHtml(String(e.message || e)) + "</div>";
  }
}

function render(data) {
  const parts = [];
  parts.push("<div class='meta'>共约 <b>" + data.total_docs.toLocaleString() +
    "</b> 篇文献，返回 <b>" + data.results.length + "</b> 条 ｜ 用时 <b>" +
    data.elapsed_ms + " ms</b> ｜ 引擎：" + data.label + "</div>");
  if (!data.results.length) {
    parts.push("<div class='empty'>没有找到相关文献，换个关键词试试（例如上面的示例查询）</div>");
  }
  const maxScore = Math.max(...data.results.map(r => r.score), 1e-9);
  data.results.forEach((r, i) => {
    parts.push(
      "<div class='result'>" +
        "<div class='head'>" +
          "<span class='rank'>" + (i + 1) + "</span>" +
          "<div class='scorebar'><i style='width:" + Math.max(3, 100 * r.score / maxScore) + "%'></i></div>" +
          "<span class='score'>" + r.score.toFixed(4) + "</span>" +
        "</div>" +
        "<div class='title'>" + highlight(r.text, r.tokens) + "</div>" +
        "<div class='docid'>" + r.doc_id + "</div>" +
      "</div>");
  });
  $("output").innerHTML = parts.join("");
}

function resetHome() {
  $("hero").style.display = "block";
  $("output").style.display = "none";
  $("q").value = "";
  $("q").focus();
}
</script>
</body>
</html>
"""


# ================================================================ 引擎实现
class Bm25Engine(object):
    """BM25 关键词召回: 纯 CPU, 秒级建索引。"""

    name = "bm25"
    label = "BM25 关键词召回"

    def __init__(self, args):
        t0 = time.time()
        self.index = BM25Index.build(args.corpus, k1=1.5, b=0.75, verbose=False)
        self.doc_texts = dict(zip(self.index.doc_ids, load_corpus(args.corpus)[1]))
        self.total_docs = self.index.n_docs
        self.load_seconds = time.time() - t0

    def search(self, query, topk):
        t0 = time.time()
        hits = self.index.search(query, topk)
        results = [{
            "doc_id": did,
            "text": self.doc_texts.get(did, ""),
            "score": round(score, 4),
            "tokens": tokens,
        } for did, score, tokens in hits]
        return results, round((time.time() - t0) * 1000, 1)


class ErnieEngine(object):
    """ERNIE 双塔 + HNSW 向量召回: 需要 paddle 与离线构建的 index/。"""

    name = "ernie"
    label = "ERNIE 双塔 + HNSW 向量召回"

    def __init__(self, args):
        import paddle
        paddle.set_device(args.device)
        from paddlenlp.transformers import AutoModel, AutoTokenizer
        from src.models import SemanticIndexBase
        from src.ann import load_index

        self.doc_ids, self.doc_texts = load_corpus(args.corpus)
        self.total_docs = len(self.doc_ids)

        id_map_path = os.path.join(args.index_dir, "doc_ids.json")
        self.row_ids = None
        if os.path.isfile(id_map_path):
            with open(id_map_path, "r", encoding="utf-8") as f:
                self.row_ids = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        pretrained = AutoModel.from_pretrained(args.base_model)
        self.model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)
        if args.weights and os.path.isfile(args.weights):
            self.model.set_dict(paddle.load(args.weights))
        self.model.eval()
        self.index = load_index(os.path.join(args.index_dir, "final_index.bin"),
                                dim=args.output_emb_size, ef=args.ef)
        self.load_seconds = 0.0

    def search(self, query, topk):
        import paddle
        t0 = time.time()
        encoded = self.tokenizer(text=query, max_seq_len=60)
        embedding = self.model.get_pooled_embedding(
            paddle.to_tensor([encoded["input_ids"]]),
            token_type_ids=paddle.to_tensor([encoded["token_type_ids"]]))
        recalled, distances = self.index.knn_query(embedding.numpy(), topk)
        results = []
        for row, dist in zip(recalled[0], distances[0]):
            did = self.row_ids[row] if self.row_ids else "line{}".format(row)
            results.append({
                "doc_id": did,
                "text": self.doc_texts.get(did, ""),
                "score": round(1.0 - float(dist), 4),
                "tokens": tokenize(query),
            })
        return results, round((time.time() - t0) * 1000, 1)


class RerankEngine(object):
    """BM25 召回 + 交叉编码器精排(两阶段完整流水线)。"""

    name = "rerank"
    label = "BM25 召回 + ERNIE-3.0 精排"

    def __init__(self, args):
        self.base = Bm25Engine(args)
        self.candidates = max(args.topk_max, 200)
        self.total_docs = self.base.total_docs

        import paddle
        self.paddle = paddle
        paddle.set_device(args.device)
        from paddlenlp.transformers import AutoModel, AutoTokenizer
        from src.models import PairwiseMatching

        self.rank_tokenizer = AutoTokenizer.from_pretrained(args.rank_model)
        rank_pretrained = AutoModel.from_pretrained(args.rank_model)
        self.rank_model = PairwiseMatching(rank_pretrained)
        self.rank_model.set_dict(paddle.load(args.rank_weights))
        self.rank_model.eval()
        self.load_seconds = 0.0

    def search(self, query, topk):
        t0 = time.time()
        paddle = self.paddle
        hits = self.base.index.search(query, self.candidates)
        if not hits:
            return [], 0.0

        seqs = []
        for did, _bm25, _tokens in hits:
            encoded = self.rank_tokenizer(
                text=query, text_pair=self.base.doc_texts.get(did, ""),
                max_seq_len=128)
            seqs.append((encoded["input_ids"], encoded["token_type_ids"]))

        probs = []
        batch = 128
        with paddle.no_grad():
            for b0 in range(0, len(seqs), batch):
                part = seqs[b0:b0 + batch]
                L = max(len(s[0]) for s in part)
                ids = np.full((len(part), L), self.rank_tokenizer.pad_token_id, dtype="int64")
                types = np.zeros((len(part), L), dtype="int64")
                for i, (rid, rtype) in enumerate(part):
                    ids[i, :len(rid)] = rid
                    types[i, :len(rtype)] = rtype
                p = self.rank_model.predict(
                    input_ids=paddle.to_tensor(ids),
                    token_type_ids=paddle.to_tensor(types))
                probs.append(p.numpy()[:, 0])
        probs = np.concatenate(probs)

        q_tokens = hits[0][2]
        scored = sorted(zip(hits, probs), key=lambda x: -x[1])[:topk]
        results = [{
            "doc_id": did,
            "text": self.base.doc_texts.get(did, ""),
            "score": round(float(p), 4),
            "tokens": q_tokens,
        } for (did, _bm25, _tokens), p in scored]
        return results, round((time.time() - t0) * 1000, 1)


# ================================================================ HTTP 服务
class Handler(BaseHTTPRequestHandler):
    engine = None       # 由 main() 注入
    page_html = EMBEDDED_HTML

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       Handler.page_html.encode("utf-8"))
        elif parsed.path == "/api/health":
            info = {
                "engine": self.engine.name,
                "label": self.engine.label,
                "docs": self.engine.total_docs,
                "load_seconds": round(getattr(self.engine, "load_seconds", 0.0), 1),
            }
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(info, ensure_ascii=False).encode("utf-8"))
        elif parsed.path == "/api/search":
            qs = parse_qs(parsed.query)
            query = (qs.get("q") or [""])[0].strip()
            try:
                topk = max(1, min(100, int((qs.get("topk") or ["10"])[0])))
            except ValueError:
                topk = 10
            if not query:
                self._send(400, "application/json; charset=utf-8",
                           json.dumps({"error": "empty query"}).encode("utf-8"))
                return
            try:
                results, elapsed_ms = self.engine.search(query, topk)
                body = json.dumps({
                    "query": query, "engine": self.engine.name,
                    "label": self.engine.label, "elapsed_ms": elapsed_ms,
                    "total_docs": self.engine.total_docs,
                    "results": results,
                }, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            except Exception as exc:  # 单次查询异常不拖垮服务
                self._send(500, "application/json; charset=utf-8",
                           json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def log_message(self, fmt, *log_args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % log_args))


def main(args):
    engines = {"bm25": Bm25Engine, "ernie": ErnieEngine, "rerank": RerankEngine}
    if args.engine not in engines:
        sys.exit("未知引擎: {} (可选: bm25/ernie/rerank)".format(args.engine))
    print("加载引擎 {} ...".format(args.engine), flush=True)
    Handler.engine = engines[args.engine](args)
    print("引擎就绪: {} (加载耗时 {:.1f}s)".format(
        Handler.engine.label, getattr(Handler.engine, "load_seconds", 0.0)), flush=True)

    url = "http://{}:{}".format(args.host, args.port)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("搜索服务已启动: {}  (Ctrl+C 退出)".format(url), flush=True)
    if args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="bm25", choices=["bm25", "ernie", "rerank"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--no-browser", dest="open_browser", action="store_false",
                    help="启动后不自动打开浏览器")
    # ernie / rerank 引擎参数
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index"))
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--rank-model", default="ernie-3.0-medium-zh")
    ap.add_argument("--rank-weights", default=os.path.join(ROOT, "weights", "rank", "rank_model_state.pdparams"))
    ap.add_argument("--topk-max", type=int, default=200, help="精排引擎的最大候选数")
    main(ap.parse_args())
