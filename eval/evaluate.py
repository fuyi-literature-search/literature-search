# -*- coding: utf-8 -*-
"""统一评估入口: 读取 run 文件与 qrels, 输出 Recall@N / MRR@10 / Hit@10 / nDCG@10。

run 文件格式(TSV, UTF-8):  查询 \t doc_id \t 分数
  每行一条召回结果; 同一查询的行不必连续, 评估器按分数重排并按 doc_id 去重(保留最优名次)。
qrels 格式(TSV):  查询 \t doc_id   (每行一个正例, 由 scripts/build_qrels.py 生成)

用法:
    python eval/evaluate.py --qrels data/qrels_dev.tsv --run runs/ernie_dev.run.tsv --recall-ks 50 100 200
    python eval/evaluate.py --qrels ... --run runs/rrf_dev.run.tsv --baseline runs/ernie_dev.run.tsv
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_run, paired_bootstrap
from src.utils import normalize


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            q, did = parts[0], parts[1]
            if q and did:
                qrels[normalize(q)].add(did)
    return qrels


def load_run(path, topk=None):
    runs = defaultdict(list)  # 规范化查询 -> [(score, doc_id)]
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
        if topk:
            ranked = ranked[:topk]
        out[q] = ranked
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=None, help="可选, 对照 run, 输出配对差值与置信区间")
    ap.add_argument("--topk", type=int, default=0,
                    help="每查询只取分数最高的前 topk 条再评估(0=全部); 精排 N=50/100/200 对照用")
    ap.add_argument("--recall-ks", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--mrr-k", type=int, default=10)
    ap.add_argument("--hit-k", type=int, default=10)
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--out", default=None, help="可选, 结果另存为 JSON")
    ap.add_argument("--note", default=None, help="可选, 写入 JSON 的口径/配置说明")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run, topk=args.topk or None)

    agg, per_query = evaluate_run(
        run, qrels, ks=tuple(args.recall_ks), mrr_k=args.mrr_k,
        hit_k=args.hit_k, ndcg_k=args.ndcg_k)

    print("=== 评估结果: {} ===".format(os.path.basename(args.run)))
    print("可评价查询数: {} / qrels 中查询数: {}".format(agg.get("n_queries", 0), len(qrels)))
    missing = len(qrels) - agg.get("n_queries", 0)
    if missing > 0:
        print("警告: {} 个 qrels 查询在 run 中无任何结果(已排除, 请检查召回覆盖)".format(missing))
    metric_names = [k for k in agg if k != "n_queries"]
    for name in metric_names:
        print("{:>12}: {:.4f}".format(name, agg[name]))

    result = {"run": args.run, "aggregate": agg}
    if args.note:
        result["note"] = args.note

    if args.baseline:
        base = load_run(args.baseline, topk=args.topk or None)
        agg_b, per_base = evaluate_run(
            base, qrels, ks=tuple(args.recall_ks), mrr_k=args.mrr_k,
            hit_k=args.hit_k, ndcg_k=args.ndcg_k)
        print("\n=== 对照: {} ===".format(os.path.basename(args.baseline)))
        for name in metric_names:
            print("{:>12}: {:.4f}".format(name, agg_b.get(name, 0.0)))
        print("\n=== 配对差值(run - baseline) 与 95% bootstrap 置信区间 ===")
        result["baseline"] = {"run": args.baseline, "aggregate": agg_b}
        result["paired"] = {}
        for name in metric_names:
            if name not in agg or name not in agg_b:
                continue
            md, lo, hi, n = paired_bootstrap(per_query, per_base, name)
            sig = "" if (lo <= 0 <= hi) else "  [*显著]"
            print("{:>12}: {:+.4f}  [{:+.4f}, {:+.4f}]  n={}{}".format(name, md, lo, hi, n, sig))
            result["paired"][name] = {"mean_diff": md, "ci_low": lo, "ci_high": hi}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
