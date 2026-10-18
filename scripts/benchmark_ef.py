# -*- coding: utf-8 -*-
"""ef 参数实验(开题: "先固定 HNSW 参数比较, 再单独调整查询 ef, 以精确向量 Top-K 评价近邻检索误差")。

对前 --n-queries 条 dev 查询编码, 以 ef=1000 为近似精确基准,
测各 ef 的 Top-K 命中率(与基准的重合率)与检索延迟, 输出 runs/bench_ef.json。

用法:
    python scripts/benchmark_ef.py --device gpu
"""
import argparse
import json
import os
import sys
import time
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

from src.models import SemanticIndexBase
from src.data import convert_bi_example, make_batchify, create_dataloader, read_queries
from src.ann import load_index


def main(args):
    paddle.set_device(args.device)
    index = load_index(os.path.join(args.index_dir, "final_index.bin"), dim=args.output_emb_size, ef=1000)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)
    model.set_dict(paddle.load(args.weights))

    queries = [q["query"] for q in read_queries(args.queries)][: args.n_queries]
    trans = partial(convert_bi_example, tokenizer=tokenizer, max_seq_length=60)
    ds = MapDataset([{"query": q} for q in queries])
    loader = create_dataloader(ds, mode="infer", batch_size=64,
                               batchify_fn=make_batchify("bi_infer", tokenizer), trans_fn=trans)
    embs = []
    with paddle.no_grad():
        for b in model.get_semantic_embedding(loader):
            embs.append(b.numpy())
    embs = np.concatenate(embs, axis=0)
    print("encoded:", embs.shape)

    K = args.k
    EFS = [1000, 50, 100, 200, 400]     # 先算 ef=1000 作为近似精确基准
    results = {}
    ref_sets = None
    for ef in EFS:
        index.set_ef(ef)
        t0 = time.time()
        labels, _ = index.knn_query(embs, K)
        dt = (time.time() - t0) / len(embs) * 1000
        sets = [set(l.tolist()) for l in labels]
        if ef == 1000:
            ref_sets = sets
            overlap = 1.0
        else:
            overlap = np.mean([len(a & b) / float(K) for a, b in zip(sets, ref_sets)])
        results[ef] = {"ms_per_query": round(dt, 2), "top{}_overlap_vs_ef1000".format(K): round(float(overlap), 4)}
        print("ef={:>4}: {:.2f} ms/query | top{} overlap vs ef=1000: {:.4f}".format(ef, dt, K, overlap))

    out = os.path.join(ROOT, "runs", "bench_ef.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"k": K, "n_queries": len(embs), "results": results}, f, ensure_ascii=False, indent=2)
    print("written:", out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index"))
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=200)
    main(ap.parse_args())
