# -*- coding: utf-8 -*-
"""RRF (Reciprocal Rank Fusion) 多路排名融合。

把两路(如 BM25 与 ERNIE)的 run 文件按 doc_id 融合:
    S(d) = Σ_r 1[d ∈ L_r] / (c + rank_r(d))
  - 每路先按分数取前 --depth 名再参与融合(固定每路深度, 保证公平比较)
  - rank 从 1 开始; 某路未召回该文献时贡献为 0; c 默认 60
  - 融合结果每查询截取前 --topn 条

用法:
    python eval/rrf.py --runs runs/bm25_dev.run.tsv runs/ernie_dev.run.tsv \
        --depth 200 --c 60 --topn 200 --out runs/rrf_dev.run.tsv
"""
import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import normalize


def load_top(path, depth):
    """加载 run 文件, 每查询按分数取前 depth 名, 并按 doc_id 去重(保留最优名次)。"""
    runs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            q, did, score = parts[0], parts[1], float(parts[2])
            if q and did:
                runs[normalize(q)].append((score, did))
    out = {}
    for q, items in runs.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        seen, ranked = set(), []
        for _, did in items:
            if did in seen:
                continue
            seen.add(did)
            ranked.append(did)
        out[q] = ranked[:depth]
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="两个及以上 run 文件")
    ap.add_argument("--depth", type=int, default=200, help="每路参与融合的深度")
    ap.add_argument("--c", type=float, default=60.0, help="RRF 常数 c")
    ap.add_argument("--topn", type=int, default=200, help="融合后每查询保留条数")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_runs = [load_top(p, args.depth) for p in args.runs]

    queries = set()
    for r in all_runs:
        queries.update(r.keys())

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for q in sorted(queries):
            scores = defaultdict(float)
            for r in all_runs:
                for rank, did in enumerate(r.get(q, []), start=1):
                    scores[did] += 1.0 / (args.c + rank)
            top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[: args.topn]
            for did, s in top:
                f.write("{}\t{}\t{:.6f}\n".format(q, did, s))
                n_written += 1

    print("queries fused: {} | lines: {}".format(len(queries), n_written))
    print("written:", args.out)


if __name__ == "__main__":
    main()
