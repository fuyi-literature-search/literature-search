# -*- coding: utf-8 -*-
"""两路召回并集覆盖率分析(开题: "统计两路并集覆盖率与融合后的 Recall@N, 区分互补收益和截断损失")。

对两路 run 文件各取前 --topk 名, 统计:
  - 各路 Recall@topk 与 并集 Recall@topk
  - 互补收益 = 并集 - 最强单路; 融合截断损失 = 融合 run - 并集

用法:
    python scripts/union_coverage.py --runs runs/bm25_dev.run.tsv runs/ernie_dev.run.tsv \
        --fusion runs/rrf_dev.run.tsv --topk 200 --out runs/union_coverage.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from evaluate import load_qrels, load_run  # noqa: E402


def main(args):
    qrels = load_qrels(args.qrels)
    arms = {os.path.basename(p): load_run(p, topk=args.topk) for p in args.runs}
    fusion = load_run(args.fusion, topk=args.topk) if args.fusion else None

    names = list(arms)
    common = [q for q in qrels if all(q in arms[n] for n in names)]
    if fusion:
        common = [q for q in common if q in fusion]
    total = sum(len(qrels[q]) for q in common)

    hits = {}
    for n in names:
        hits[n] = sum(len(set(arms[n][q]) & qrels[q]) for q in common)
    union_hits = sum(len(set().union(*[set(arms[n][q]) for n in names]) & qrels[q]) for q in common)
    fus_hits = sum(len(set(fusion[q]) & qrels[q]) for q in common) if fusion else None

    best = max(hits.values())
    result = {
        "topk": args.topk, "n_queries": len(common),
        "recall@{}".format(args.topk): {n: round(h / total, 4) for n, h in hits.items()},
        "union": round(union_hits / total, 4),
        "互补收益(并集-最强单路)": round(union_hits / total - best / total, 4),
    }
    if fusion:
        result["融合截断损失(融合-并集)"] = round(fus_hits / total - union_hits / total, 4)
        result["recall@{}[fusion]".format(args.topk)] = round(fus_hits / total, 4)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("written:", args.out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrels", default=os.path.join(ROOT, "data", "qrels_dev.tsv"))
    ap.add_argument("--runs", nargs=2, required=True, help="两个单路 run 文件")
    ap.add_argument("--fusion", default=None, help="可选, 融合 run 文件")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "union_coverage.json"))
    main(ap.parse_args())
