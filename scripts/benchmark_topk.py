# -*- coding: utf-8 -*-
"""步骤 9B: 召回层延迟随 topk 的变化基准——检验"召回集大小对延迟的影响"。

对同一批查询(--n 条)分别测:
  - BM25 词级 / 字符 bigram(CPU): index.search(query, k) 逐 k 计时
  - ERNIE 双塔(GPU): 查询编码一次(topk 无关的常量) + knn_query 逐 k 计时
    (hnswlib 要求 ef >= k, 各 k 取 ef = max(base_ef, 1.2k), 如实报告)
  - 精排折算: 引用 runs/bench_rank.json 的实测吞吐(对/秒)线性折算, 非实测

输出 JSON 供绘图(runs/bench_topk.json)。

用法:
    python scripts/benchmark_topk.py
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KS = [10, 50, 100, 200, 500, 1000]


def bench(fn, n):
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - t0) * 1000.0 / n


def main(args):
    from src.bm25 import BM25Index, _TOKENIZERS
    from src.utils import read_queries

    result = {"ks": KS, "n_queries": args.n, "device_ernie": args.device, "per_k": {}}
    queries = [q["query"] for q in read_queries(args.queries)[: args.n]]
    print("查询数:", len(queries))

    # ---------- BM25 词级 / 字符 ----------
    for name, tok in (("bm25_word", None), ("bm25_char", _TOKENIZERS["char"])):
        t0 = time.time()
        index = BM25Index.build(args.corpus, tokenizer=tok)
        print("构建 {} 索引: {:.0f}s".format(name, time.time() - t0), flush=True)
        per_k = {}
        for k in KS:
            per_k[k] = round(bench(lambda i, k=k: index.search(queries[i], k), len(queries)), 2)
        result["per_k"][name] = per_k
        print(name, per_k)

    # ---------- ERNIE(GPU): 编码一次 + 逐 k 检索 ----------
    import numpy as np
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
        json.load(f)  # 仅验证侧车存在

    t0 = time.perf_counter()
    encoded = [tokenizer(text=q, max_seq_len=args.max_seq_len) for q in queries]
    embs = []
    with paddle.no_grad():
        for i in range(0, len(encoded), 32):
            batch = encoded[i:i + 32]
            L = max(len(e["input_ids"]) for e in batch)
            ids = np.full((len(batch), L), tokenizer.pad_token_id, dtype="int64")
            types = np.zeros((len(batch), L), dtype="int64")
            for r, e in enumerate(batch):
                ids[r, :len(e["input_ids"])] = e["input_ids"]
                types[r, :len(e["token_type_ids"])] = e["token_type_ids"]
            out = model.get_pooled_embedding(paddle.to_tensor(ids), token_type_ids=paddle.to_tensor(types))
            embs.append(out.numpy())
    embs = np.concatenate(embs, axis=0)
    result["ernie_encode_ms"] = round((time.perf_counter() - t0) * 1000.0 / len(queries), 2)
    print("ERNIE 编码(常量, 与 topk 无关): {:.2f} ms/查询".format(result["ernie_encode_ms"]))

    per_k, ef_used = {}, {}
    for k in KS:
        ef = max(args.ef, int(1.2 * k))
        index = load_index(os.path.join(args.index_dir, "final_index.bin"),
                           dim=args.output_emb_size, ef=ef)
        per_k[k] = round(bench(lambda i: index.knn_query(embs[i:i + 1], k), len(queries)), 2)
        ef_used[k] = ef
    result["per_k"]["ernie_knn"] = per_k
    result["ef_used"] = ef_used
    print("ernie_knn", per_k, "| ef:", ef_used)

    # ---------- 精排折算(引用实测吞吐) ----------
    rank_path = os.path.join(ROOT, "runs", "bench_rank.json")
    if os.path.isfile(rank_path):
        with open(rank_path, "r", encoding="utf-8") as f:
            ms_per_pair = json.load(f)["ms_per_pair"]
        result["rerank_ms_per_pair"] = ms_per_pair
        result["per_k"]["pipeline_bm25_plus_rerank"] = {
            k: round(result["per_k"]["bm25_word"][k] + k * ms_per_pair, 1) for k in KS}
        result["per_k"]["pipeline_ernie_plus_rerank"] = {
            k: round(result["ernie_encode_ms"] + per_k[k] + k * ms_per_pair, 1) for k in KS}
        print("rerank 折算: {:.2f} ms/对 (bench_rank 实测吞吐)".format(ms_per_pair))

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
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "bench_topk.json"))
    main(ap.parse_args())
