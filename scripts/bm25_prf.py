# -*- coding: utf-8 -*-
"""步骤 5D/8: BM25 + 伪相关反馈(PRF)召回路——批量生成 run 文件。

两轮检索:
  1. BM25 初检, 取前 fb_docs 篇作为伪相关文档集
  2. 从伪相关文档的 token 中按 idf × 反馈文档频 选出 fb_terms 个扩展词
     (排除原查询词), 追加进查询再做一轮 BM25, 取 top-200

作为词法家族的"增强版"对照, 用来回答"不换语义模型, 仅靠查询扩展
能把词法召回推到什么水平"。扩展词与原查询词在打分中同权(各计一次)。

用法:
    python scripts/bm25_prf.py --limit 200        # 快速验证前 200 条
    python scripts/bm25_prf.py                    # dev 全量
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bm25 import BM25Index, tokenize
from src.utils import read_queries


def expand_tokens(index, doc_texts, query, fb_docs, fb_terms):
    """返回扩展词列表。词分 = idf × 出现该词的反馈文档数, 排除原查询词。"""
    q_tokens = set(tokenize(query))
    fb = index.search(query, fb_docs)
    if not fb:
        return []
    df = Counter()
    for did, _score, _t in fb:
        df.update(set(tokenize(doc_texts.get(did, ""))))
    ranked = sorted(
        ((tok, n) for tok, n in df.items() if tok not in q_tokens and tok in index.idf),
        key=lambda x: (-index.idf[x[0]] * x[1], x[0]))
    return [tok for tok, _ in ranked[:fb_terms]]


def main(args):
    t0 = time.time()
    index = BM25Index.build(args.corpus, k1=args.k1, b=args.b)
    print("index built in {:.1f}s".format(time.time() - t0))

    doc_texts = {}
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                doc_texts[parts[0]] = parts[1]
    print("doc texts loaded:", len(doc_texts))

    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print("查询数:", len(queries))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(queries):
            extra = expand_tokens(index, doc_texts, item["query"], args.fb_docs, args.fb_terms)
            for did, score, _tokens in index.search(item["query"], args.topk, extra_tokens=extra):
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
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"),
                    help="必须与其他召回路使用同一清洗后语料")
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--topk", type=int, default=200,
                    help="每查询返回条数; 对照实验需要 N=50/100/200, 故默认 200")
    ap.add_argument("--k1", type=float, default=1.5, help="BM25 k1")
    ap.add_argument("--b", type=float, default=0.75, help="BM25 b")
    ap.add_argument("--fb-docs", type=int, default=10, help="第一轮取前 N 篇作伪相关文档")
    ap.add_argument("--fb-terms", type=int, default=10, help="扩展词个数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条查询, 0 表示全部")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "bm25_prf_dev.run.tsv"))
    main(ap.parse_args())
