# -*- coding: utf-8 -*-
"""五路召回对照实验汇总: 统一加载 run 文件, 计算指标并生成 Markdown 汇总表。

输入为统一 run 文件(查询 \t doc_id \t 分数), 全部截断 top-200 后用同一份
qrels 评测(与 eval/evaluate.py 完全同一口径), 相对基线(BM25)输出配对
bootstrap 95% 置信区间, 结果写入 runs/recall_comparison.md。

用法:
    python scripts/summarize_recall.py
    python scripts/summarize_recall.py --baseline tfidf   # 换配对基线
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from evaluate import load_qrels, load_run          # noqa: E402
from metrics import evaluate_run, paired_bootstrap  # noqa: E402

ARMS = [
    ("TF-IDF", os.path.join(ROOT, "runs", "tfidf_dev.run.tsv")),
    ("BM25", os.path.join(ROOT, "runs", "bm25_dev.run.tsv")),
    ("字符bigram BM25", os.path.join(ROOT, "runs", "charbm25_dev.run.tsv")),
    ("QLM(μ=3000)", os.path.join(ROOT, "runs", "qlm_dev.run.tsv")),
    ("BM25+PRF", os.path.join(ROOT, "runs", "bm25_prf_dev.run.tsv")),
    ("ERNIE 零训练", os.path.join(ROOT, "runs", "ernie_zeroshot_dev.run.tsv")),
    ("ERNIE 微调双塔", os.path.join(ROOT, "runs", "ernie_dev.run.tsv")),
    ("RRF(BM25+ERNIE)", os.path.join(ROOT, "runs", "rrf2_bm25_ernie_dev.run.tsv")),
    ("RRF三路(+字符)", os.path.join(ROOT, "runs", "rrf3_dev.run.tsv")),
]
KS = (10, 50, 100, 200)


def fmt(x):
    return "{:.2f}".format(x * 100.0)


def main(args):
    qrels = load_qrels(args.qrels)
    print("qrels 查询数:", len(qrels))

    aggs, per_qs = {}, {}
    for label, path in ARMS:
        if not os.path.isfile(path):
            print("跳过(缺 run 文件):", label, "->", path)
            continue
        t0 = time.time()
        run = load_run(path, topk=args.topk)
        agg, per_q = evaluate_run(run, qrels, ks=KS)
        aggs[label], per_qs[label] = agg, per_q
        print("{:<14} n={:<6} ({:.0f}s)".format(label, agg.get("n_queries", 0), time.time() - t0))

    baseline = args.baseline if args.baseline in per_qs else "BM25"
    cols = ["hit@10", "mrr@10", "ndcg@10"] + ["recall@{}".format(k) for k in KS]

    lines = ["# 召回方法对照实验(dev, 同一清洗语料 29.2 万, 截断 top-{})".format(args.topk), ""]
    lines.append("| 召回路 | " + " | ".join(
        ["Hit@10", "MRR@10", "nDCG@10"] + ["R@{}".format(k) for k in KS]) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for label, _ in ARMS:
        if label not in aggs:
            continue
        a = aggs[label]
        lines.append("| {} | ".format(label) + " | ".join(fmt(a[c]) for c in cols) + " |")

    lines += ["", "## 相对基线 {} 的配对差值(百分点, bootstrap 95% CI)".format(baseline), ""]
    lines.append("| 召回路 | ΔR@200 [95% CI] | ΔMRR@10 [95% CI] | 显著(两区间均不含 0) |")
    lines.append("|---|---|---|---|")
    for label, _ in ARMS:
        if label not in per_qs or label == baseline:
            continue
        cells = []
        sig = True
        for metric in ("recall@200", "mrr@10"):
            diff, lo, hi, n = paired_bootstrap(per_qs[label], per_qs[baseline], metric)
            cells.append("{:+.2f} [{:+.2f}, {:+.2f}]".format(
                diff * 100, lo * 100, hi * 100))
            sig = sig and (lo > 0 or hi < 0)
        lines.append("| {} | {} | {} | {} |".format(
            label, cells[0], cells[1], "是" if sig else "否"))

    lines += ["", "## 口径说明", "",
              "- 各路同一清洗语料(29.2 万篇)独立建索引, 同一份 dev qrels, 统一截断 top-200。",
              "- 正例多为单一精确标题匹配, R@10 数值上等于 Hit@10。",
              "- 词法各路(TF-IDF/BM25/字符/QLM/PRF)约 30 条查询无命中词不产生任何候选, 计入 n 的差异(<0.2%), 未按 0 分插补。",
              "- QLM 平滑强度 μ 在 500 条查询子集上从 {10,50,200,500,1000,3000} 中选定; 短标题语料需远大于常规长文档的平滑。",
              "- ERNIE 零训练 = 底座不微调直接编码(含随机初始化的输出投影层), 其 R@200 与"
              "29.2 万篇中随机取 200 篇的命中期望同量级, 即检索能力接近随机。",
              "- RRF 融合 c=5(沿用已调参数), depth=200; 融合臂与单路同口径评测。"]

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("written:", args.out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrels", default=os.path.join(ROOT, "data", "qrels_dev.tsv"))
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--baseline", default="BM25", help="配对差值的基线(表内标签名)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "recall_comparison.md"))
    main(ap.parse_args())
