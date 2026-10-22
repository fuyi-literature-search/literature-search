# -*- coding: utf-8 -*-
"""步骤 5F/8: 查询似然模型 QLM 召回路(词法对照基线)——批量生成 run 文件。

语言模型检索的经典范式: 把文档看作一元语言模型, 查询由其生成的概率
    score(d) = Σ_{t∈q} qtf(t) · log p(t|d)
    p(t|d) = (tf(t,d) + μ·p(t|C)) / (|d| + μ)      (Dirichlet 平滑)
p(t|C) 为词 t 的集合概率, μ 为平滑强度。与 TF-IDF(向量加权)/BM25(饱和+
长度归一)构成词法检索三大范式的完整对照。

复用 src/bm25.py 的倒排结构(postings/doc_len), 仅换打分公式。

用法:
    python scripts/qlm_runner.py --limit 200       # 快速验证
    python scripts/qlm_runner.py                   # dev 全量
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bm25 import BM25Index
from src.utils import read_queries


def main(args):
    t0 = time.time()
    index = BM25Index.build(args.corpus)
    # 集合级一元概率 p(t|C)
    total_tf = {t: float(p[1].sum()) for t, p in index.postings.items()}
    total_len = float(index.doc_len.sum())
    p_c = {t: v / total_len for t, v in total_tf.items()}
    print("index built in {:.1f}s | vocab: {}".format(time.time() - t0, len(p_c)))

    mu = args.mu

    def search(query, topk):
        qtf = Counter(t for t in index.tokenize(query) if t in index.postings)
        if not qtf:
            return []
        idx_parts, sc_parts = [], []
        for tok, w in qtf.items():
            idx, tf = index.postings[tok]
            # Σ qtf·log(tf + μ·p(t|C)) 逐 posting 累加
            idx_parts.append(idx)
            sc_parts.append(w * np.log(tf + mu * p_c[tok]))
        all_idx = np.concatenate(idx_parts)
        all_sc = np.concatenate(sc_parts)
        uniq, inv = np.unique(all_idx, return_inverse=True)
        acc = np.bincount(inv, weights=all_sc)
        # 文档级长度项: (Σ qtf)·log(|d|+μ)
        scores = acc - sum(qtf.values()) * np.log(index.doc_len[uniq] + mu)
        k = min(topk, len(scores))
        sel = np.argpartition(-scores, k - 1)[:k]
        top = sorted(sel, key=lambda j: -scores[j])
        return [(index.doc_ids[uniq[j]], float(scores[j])) for j in top]

    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print("查询数:", len(queries))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(queries):
            for did, score in search(item["query"], args.topk):
                f.write("{}\t{}\t{:.6f}\n".format(item["query"], did, score))
                n_written += 1
            if (i + 1) % 2000 == 0:
                print("  {}/{} queries, {:.0f}s elapsed".format(i + 1, len(queries), time.time() - t0))

    print("queries: {} | lines: {}".format(len(queries), n_written))
    print("written:", args.output)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"))
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--mu", type=float, default=50.0,
                    help="Dirichlet 平滑强度; 语料为短标题(平均约17词), 取小于常规长文档的值")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条查询, 0 表示全部")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "qlm_dev.run.tsv"))
    main(ap.parse_args())
