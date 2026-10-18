# -*- coding: utf-8 -*-
"""步骤 9/8: 各阶段效率测试 - 延迟(P50/P95/P99)、吞吐、内存与构建耗时。

分阶段独立测量, 输出 JSON 存 runs/, 报告需注明硬件/批量/缓存条件:

  --stage bm25   BM25 路: 倒排索引构建耗时 + 内存 + 查询延迟(单线程, 含分词)
  --stage ernie  ERNIE 路: 双塔查询编码 + HNSW 检索延迟(需 index/ 与权重)
  --stage rank   精排: 交叉编码器逐对打分吞吐(需 weights/rank)

用法:
    python scripts/benchmark_latency.py --stage bm25
    python scripts/benchmark_latency.py --stage ernie --device gpu
"""
import argparse
import ctypes
import json
import os
import platform
import sys
import time
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import read_queries


def peak_rss_mb():
    """进程峰值内存(MB); 仅 Windows, 其他平台返回 None。"""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        # 句柄必须按 64 位指针传(ctypes.c_void_p(-1)), 否则伪句柄被截断导致调用失败
        fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
        if not fn(ctypes.c_void_p(-1), ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.PeakWorkingSetSize / 1048576.0, 1)
    except Exception:
        return None


def percentiles(latencies):
    s = sorted(latencies)

    def pct(p):
        return round(s[min(len(s) - 1, int(p * len(s)))] * 1000.0, 2)   # ms
    return {"mean_ms": round(sum(s) / len(s) * 1000.0, 2),
            "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
            "qps": round(len(s) / sum(s), 1), "n": len(s)}


def env_info(device="cpu"):
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "processor": platform.processor() or None, "device": device}
    try:
        import paddle
        info["paddle"] = paddle.__version__
        if device == "gpu":
            info["gpu"] = paddle.device.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def bench_bm25(args):
    from scripts.bm25_runner import BM25Index
    t0 = time.time()
    index = BM25Index.build(args.corpus)
    build_s = time.time() - t0
    print("索引构建: {:.1f}s | 峰值内存: {} MB".format(build_s, peak_rss_mb()))

    queries = [q["query"] for q in read_queries(args.queries)][: args.n_queries]

    for q in queries[:20]:                     # 预热(jieba 词典懒加载等)
        index.search(q, 10)

    lat = []
    for q in queries:
        t = time.time()
        index.search(q, args.topk)
        lat.append(time.time() - t)
    return {
        "stage": "bm25", "topk": args.topk, "threads": 1,
        "index_build_s": round(build_s, 1), "peak_rss_mb": peak_rss_mb(),
        "n_docs": index.n_docs, "avgdl": round(index.avgdl, 1),
        "vocab": len(index.postings), "k1": index.k1, "b": index.b,
        "query_latency": percentiles(lat), "env": env_info("cpu"),
    }


def bench_ernie(args):
    import paddle
    from paddlenlp.datasets import MapDataset
    from paddlenlp.transformers import AutoModel, AutoTokenizer
    from src.models import SemanticIndexBase
    from src.data import convert_bi_example, make_batchify, create_dataloader
    from src.ann import load_index

    paddle.set_device(args.device)
    index = load_index(os.path.join(args.index_dir, "final_index.bin"), dim=args.output_emb_size, ef=args.ef)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)
    model.set_dict(paddle.load(args.weights))
    model.eval()
    trans = partial(convert_bi_example, tokenizer=tokenizer, max_seq_length=60)

    queries = [q["query"] for q in read_queries(args.queries)][: args.n_queries]

    # 逐条查询(端到端交互模式): 编码 + HNSW 检索
    lat = []
    with paddle.no_grad():
        for q in queries:
            t = time.time()
            ds = MapDataset([{"query": q}])
            loader = create_dataloader(ds, mode="infer", batch_size=1,
                                       batchify_fn=make_batchify("bi_infer", tokenizer), trans_fn=trans)
            emb = next(iter(model.get_semantic_embedding(loader)))
            index.knn_query(emb.numpy(), args.topk)
            lat.append(time.time() - t)
    return {
        "stage": "ernie_recall", "topk": args.topk, "ef": args.ef,
        "mode": "single_query_e2e", "query_latency": percentiles(lat),
        "peak_rss_mb": peak_rss_mb(), "env": env_info(args.device),
    }


def bench_rank(args):
    import paddle
    from paddlenlp.datasets import MapDataset
    from paddlenlp.transformers import AutoModel, AutoTokenizer
    from src.models import PairwiseMatching
    from src.data import convert_pairwise_example, make_batchify, create_dataloader

    paddle.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = PairwiseMatching(pretrained)
    model.set_dict(paddle.load(args.weights))
    model.eval()
    trans = partial(convert_pairwise_example, tokenizer=tokenizer,
                    max_seq_length=args.max_seq_len, phase="predict")

    from src.utils import load_corpus
    _, doc_texts = load_corpus(args.corpus)
    queries = [q["query"] for q in read_queries(args.queries)][: args.n_rank_queries]
    # 每查询配 topk 个候选, 模拟精排负载
    pairs = []
    for q in queries:
        for j in range(args.topk):
            pairs.append({"query": q, "title": doc_texts[(hash(q) + j) % len(doc_texts)]})
    ds = MapDataset(pairs)
    loader = create_dataloader(ds, mode="infer", batch_size=args.batch_size,
                               batchify_fn=make_batchify("pair_predict", tokenizer), trans_fn=trans)

    @paddle.no_grad()
    def score_all():
        out = []
        for batch in loader:
            input_ids, token_type_ids = batch
            out.append(model.predict(input_ids=input_ids, token_type_ids=token_type_ids).numpy())
        return out

    score_all()                                  # 预热
    t = time.time()
    score_all()
    total_s = time.time() - t
    return {
        "stage": "rerank", "batch_size": args.batch_size, "topk_per_query": args.topk,
        "n_pairs": len(pairs), "total_s": round(total_s, 1),
        "pairs_per_s": round(len(pairs) / total_s, 1),
        "ms_per_pair": round(total_s / len(pairs) * 1000, 2),
        "peak_rss_mb": peak_rss_mb(), "env": env_info(args.device),
    }


def main(args):
    if args.stage == "bm25":
        result = bench_bm25(args)
    elif args.stage == "ernie":
        result = bench_ernie(args)
    elif args.stage == "rank":
        result = bench_rank(args)
    else:
        raise SystemExit("未知 stage: " + args.stage)

    out = os.path.join(ROOT, "runs", "bench_{}.json".format(args.stage))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("written:", out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["bm25", "ernie", "rank"])
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"))
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-queries", type=int, default=500, help="延迟测试查询数")
    ap.add_argument("--n-rank-queries", type=int, default=100, help="精排吞吐测试查询数")
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index"))
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    main(ap.parse_args())
