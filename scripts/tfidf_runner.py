# -*- coding: utf-8 -*-
"""步骤 5C/8: TF-IDF 召回路(词法对照基线)——批量生成 run 文件。

与 BM25 路(src/bm25.py)共用同一分词规则(jieba + 小写 + 去标点),
tf-idf 加权 + 余弦相似度, 输出统一 run 文件(查询 \t doc_id \t 分数)。
作为词法家族的经典对照, 用来回答"BM25 的饱和/文档长度归一化打分
相对朴素 tf-idf 加权带来多少收益"。

用法:
    python scripts/tfidf_runner.py --limit 200       # 快速验证前 200 条
    python scripts/tfidf_runner.py                   # dev 全量
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.bm25 import tokenize
from src.utils import read_queries


def build_tfidf(corpus_path):
    """返回 (vectorizer, doc_matrix csr, doc_ids)。文档行 L2 归一化, 点积即余弦。"""
    doc_ids, doc_lines = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                doc_ids.append(parts[0])
                doc_lines.append(" ".join(tokenize(parts[1])))
    # analyzer 直接按空格切(文本已预分词), 保证与 BM25 路完全同一分词
    vec = TfidfVectorizer(analyzer=str.split, token_pattern=None)
    doc_matrix = vec.fit_transform(doc_lines)     # csr, 行已 L2 归一化
    return vec, doc_matrix, doc_ids


def main(args):
    t0 = time.time()
    vec, doc_matrix, doc_ids = build_tfidf(args.corpus)
    print("docs: {} | vocab: {} | build {:.1f}s".format(
        doc_matrix.shape[0], doc_matrix.shape[1], time.time() - t0))

    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print("查询数:", len(queries))

    doc_matrix_t = doc_matrix.T.tocsc()          # V×N, 供查询向量点积
    n_docs = len(doc_ids)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(queries):
            q_vec = vec.transform([" ".join(tokenize(item["query"]))])   # 1×V, L2 归一化
            scores = q_vec.dot(doc_matrix_t).toarray().ravel()           # 余弦相似度
            k = min(args.topk, n_docs)
            if q_vec.nnz:                                                # 无命中词时全零
                sel = np.argpartition(-scores, k - 1)[:k]
                top = sorted(sel, key=lambda j: -scores[j])
                for j in top:
                    f.write("{}\t{}\t{:.6f}\n".format(item["query"], doc_ids[j], scores[j]))
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
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条查询, 0 表示全部")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "tfidf_dev.run.tsv"))
    main(ap.parse_args())
