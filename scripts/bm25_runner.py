# -*- coding: utf-8 -*-
"""步骤 5B/8: BM25 关键词召回路(双路召回之一)。

jieba 分词建立倒排索引并对查询打分, 输出与 ERNIE 路统一格式的 run 文件
(查询 \t doc_id \t 分数), 供 eval/evaluate.py 与 eval/rrf.py 直接使用。

设计约定(与开题报告一致):
  - 索引与查询使用同一分词规则: jieba 精确模式 + 小写化,
    仅保留汉字/字母/数字 token(去掉标点), 不做停用词过滤, 保留术语/缩写/数字
  - 语料与 ERNIE 路共用 data/corpus_clean.tsv(同一 doc_id 体系)
  - BM25 参数 k1/b 与统计口径全程固定, 对照实验只变召回策略与候选数 N

用法:
    python scripts/bm25_runner.py --limit 200       # 快速验证前 200 条
    python scripts/bm25_runner.py                   # dev 全量 19,927 条
"""
import argparse
import os
import re
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import jieba

from src.utils import read_queries

_TOKEN_RE = re.compile(r"^[0-9a-z\u4e00-\u9fa5]+$")


def tokenize(text):
    """索引与查询共用的分词规则: 小写 + jieba 精确模式, 去掉标点符号。"""
    return [t for t in jieba.lcut(text.lower()) if _TOKEN_RE.match(t)]


class BM25Index(object):
    """内存倒排索引。postings 用 numpy 数组存 (doc 行号, 词频), 查询向量化打分。"""

    def __init__(self, doc_ids, doc_len, postings, k1=1.5, b=0.75):
        self.doc_ids = doc_ids
        self.doc_len = doc_len                    # np.int32, 每篇 token 数
        self.postings = postings                  # token -> (idx数组, tf数组)
        self.k1, self.b = k1, b
        self.n_docs = len(doc_ids)
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0
        # Lucene 式 idf: ln(1 + (N - df + 0.5)/(df + 0.5)), 恒正
        self.idf = {
            t: float(np.log(1.0 + (self.n_docs - len(p[0]) + 0.5) / (len(p[0]) + 0.5)))
            for t, p in postings.items()
        }

    @classmethod
    def build(cls, corpus_path, k1=1.5, b=0.75, verbose=True):
        doc_ids, doc_len = [], []
        postings = defaultdict(lambda: ([], []))   # token -> (doc行号列表, tf列表)
        with open(corpus_path, "r", encoding="utf-8") as f:
            for row, line in enumerate(f):
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2 or not parts[1].strip():
                    continue
                tokens = tokenize(parts[1])
                doc_ids.append(parts[0])
                doc_len.append(len(tokens))
                for tok, tf in Counter(tokens).items():
                    postings[tok][0].append(row)
                    postings[tok][1].append(tf)
        doc_len = np.array(doc_len, dtype=np.int32)
        np_postings = {t: (np.array(li, dtype=np.int32), np.array(lt, dtype=np.float32))
                       for t, (li, lt) in postings.items()}
        if verbose:
            print("docs: {} | vocab: {} | avgdl: {:.1f}".format(
                len(doc_ids), len(np_postings), float(doc_len.mean())))
        return cls(doc_ids, doc_len, np_postings, k1=k1, b=b)

    def search(self, query, topk):
        """返回 [(doc_id, score)] 按分数降序; 查询词各计一次(标准 BM25 打分)。"""
        k1, b = self.k1, self.b
        norm = k1 * (1.0 - b + b * self.doc_len / max(self.avgdl, 1e-9))

        idx_parts, score_parts = [], []
        for tok in set(tokenize(query)):
            entry = self.postings.get(tok)
            if entry is None:
                continue
            idx, tf = entry
            denom = tf + norm[idx]
            idx_parts.append(idx)
            score_parts.append(self.idf[tok] * tf * (k1 + 1.0) / denom)
        if not idx_parts:
            return []

        all_idx = np.concatenate(idx_parts)
        all_sc = np.concatenate(score_parts)
        uniq, inv = np.unique(all_idx, return_inverse=True)
        scores = np.bincount(inv, weights=all_sc)

        k = min(topk, len(scores))
        sel = np.argpartition(-scores, k - 1)[:k]
        top = sorted(sel, key=lambda j: -scores[j])
        return [(self.doc_ids[uniq[j]], float(scores[j])) for j in top]


def main(args):
    t0 = time.time()
    index = BM25Index.build(args.corpus, k1=args.k1, b=args.b)
    print("index built in {:.1f}s".format(time.time() - t0))

    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print("查询数:", len(queries))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(queries):
            for did, score in index.search(item["query"], args.topk):
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
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条查询, 0 表示全部")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "bm25_dev.run.tsv"))
    main(ap.parse_args())
