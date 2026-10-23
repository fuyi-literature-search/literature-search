# -*- coding: utf-8 -*-
"""步骤 9C: 召回层延迟稳定性基准——四臂同机对比(换查询的开销)。

同一批查询(topk 固定), 逐查询计时, 输出延迟分布(P50/P95/P99/均值/长尾倍数):
  - BM25 词级(CPU) / 字符 bigram(CPU): index.search(q, k)
  - ERNIE 召回(GPU): 单查询端到端(逐条编码 + HNSW, 与 bench_ernie 同口径)
  - 三路 RRF 融合: 词级 + 字符 + ERNIE + 合并, 系统完整召回层

分布衡量的是"仅更换查询"带来的延迟变化(查询间方差), 与 topk 无关
(后者见 benchmark_topk.py)。四臂必须同机测得才可比。

用法:
    python scripts/benchmark_stability.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def stats(ms):
    arr = np.array(ms)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {"mean_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(p50), 2), "p95_ms": round(float(p95), 2),
            "p99_ms": round(float(p99), 2),
            "tail_ratio": round(float(p99 / max(p50, 1e-9)), 2),
            "qps": round(1000.0 / float(arr.mean()), 1), "n": len(ms)}


def main(args):
    from src.bm25 import BM25Index, _TOKENIZERS
    from src.utils import read_queries

    queries = [q["query"] for q in read_queries(args.queries)[: args.n]]
    print("查询数:", len(queries))

    # ---------- 词级 / 字符 BM25 ----------
    times = {}
    index_w = BM25Index.build(args.corpus)
    index_c = BM25Index.build(args.corpus, tokenizer=_TOKENIZERS["char"])
    for name, index in (("bm25_word", index_w), ("bm25_char", index_c)):
        ms = []
        for q in queries:
            t0 = time.perf_counter()
            index.search(q, args.topk)
            ms.append((time.perf_counter() - t0) * 1000.0)
        times[name] = ms
        print(name, stats(ms), flush=True)

    # ---------- ERNIE 单查询端到端(逐条编码, 与 bench_ernie 同口径) ----------
    import paddle
    from paddlenlp.transformers import AutoModel, AutoTokenizer
    from src.ann import load_index
    from src.models import SemanticIndexBase

    paddle.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)
    model.set_dict(paddle.load(args.weights))
    model.eval()
    with open(os.path.join(args.index_dir, "doc_ids.json"), "r", encoding="utf-8") as f:
        json.load(f)
    index = load_index(os.path.join(args.index_dir, "final_index.bin"),
                       dim=args.output_emb_size, ef=args.ef)

    def encode_one(query):
        enc = tokenizer(text=query, max_seq_len=args.max_seq_len)
        ids = np.array([enc["input_ids"]], dtype="int64")
        types = np.array([enc["token_type_ids"]], dtype="int64")
        with paddle.no_grad():
            return model.get_pooled_embedding(paddle.to_tensor(ids),
                                              token_type_ids=paddle.to_tensor(types)).numpy()

    ms_ernie, ms_fusion = [], []
    # 预热: 前几次 GPU 前向含 CUDA 内核编译等一次性开销, 不计入统计
    for q in queries[:5]:
        index.knn_query(encode_one(q), args.topk)
    for q in queries:
        # ERNIE 单路(单查询端到端)
        t0 = time.perf_counter()
        emb = encode_one(q)
        index.knn_query(emb, args.topk)
        ms_ernie.append((time.perf_counter() - t0) * 1000.0)
        # 三路融合: 词级 + 字符 + ERNIE + RRF 合并, 整体一次计时
        t0 = time.perf_counter()
        lst_w = [d for d, _s, _t in index_w.search(q, args.topk)]
        lst_c = [d for d, _s, _t in index_c.search(q, args.topk)]
        emb2 = encode_one(q)
        lst_e = [str(r) for r in index.knn_query(emb2, args.topk)[0][0]]
        scores = {}
        for lst in (lst_w, lst_c, lst_e):
            for r, did in enumerate(lst, 1):
                scores[did] = scores.get(did, 0.0) + 1.0 / (5.0 + r)
        sorted(scores.items(), key=lambda x: -x[1])[: args.topk]
        ms_fusion.append((time.perf_counter() - t0) * 1000.0)
    times["ernie_e2e"] = ms_ernie
    times["rrf3_fusion"] = ms_fusion
    print("ernie_e2e", stats(ms_ernie), flush=True)
    print("rrf3_fusion", stats(ms_fusion), flush=True)

    result = {"topk": args.topk, "device_ernie": args.device, "arms": {
        k: stats(v) for k, v in times.items()}}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("written:", args.out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"))
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index_finetuned_clean"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=60)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "bench_stability.json"))
    main(ap.parse_args())
