# -*- coding: utf-8 -*-
"""步骤 5B/8: BM25 关键词召回路(双路召回之一)——批量生成 run 文件。

核心实现在 src/bm25.py(与 webapp/server.py 在线搜索共用), 本脚本只做批处理封装:
jieba 分词建立倒排索引并对查询打分, 输出与 ERNIE 路统一格式的 run 文件
(查询 \t doc_id \t 分数), 供 eval/evaluate.py 与 eval/rrf.py 直接使用。

用法:
    python scripts/bm25_runner.py --limit 200       # 快速验证前 200 条
    python scripts/bm25_runner.py                   # dev 全量 19,927 条
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bm25 import BM25Index, _TOKENIZERS
from src.utils import read_queries


def main(args):
    t0 = time.time()
    index = BM25Index.build(args.corpus, k1=args.k1, b=args.b,
                            tokenizer=_TOKENIZERS[args.tokenizer])
    print("tokenizer: {} | index built in {:.1f}s".format(args.tokenizer, time.time() - t0))

    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print("查询数:", len(queries))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(queries):
            for did, score, _tokens in index.search(item["query"], args.topk):
                f.write("{}\t{}\t{:.6f}\n".format(item["query"], did, score))
                n_written += 1
            if (i + 1) % 2000 == 0:
                print("  {}/{} queries, {:.0f}s elapsed".format(i + 1, len(queries), time.time() - t0))

    print("queries: {} | lines: {}".format(len(queries), n_written))
    print("written: {}".format(args.output))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"),
                    help="必须与 ERNIE 路使用同一清洗后语料")
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--topk", type=int, default=200,
                    help="每查询返回条数; 对照实验需要 N=50/100/200, 故默认 200")
    ap.add_argument("--k1", type=float, default=1.5, help="BM25 k1")
    ap.add_argument("--b", type=float, default=0.75, help="BM25 b")
    ap.add_argument("--tokenizer", default="word", choices=["word", "char"],
                    help="word=jieba 词级(默认); char=字符二元组(分词消融对照)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条查询, 0 表示全部")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "bm25_dev.run.tsv"))
    main(ap.parse_args())
