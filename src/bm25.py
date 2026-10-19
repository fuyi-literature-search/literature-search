# -*- coding: utf-8 -*-
"""BM25 关键词召回核心(倒排索引 + BM25 打分)。

被 scripts/bm25_runner.py(批量召回)与 webapp/server.py(在线搜索服务)共用。

设计约定(与开题报告一致):
  - 索引与查询使用同一分词规则: jieba 精确模式 + 小写化,
    仅保留汉字/字母/数字 token(去掉标点), 不做停用词过滤, 保留术语/缩写/数字
  - 语料为 data/corpus_clean.tsv, 与 ERNIE 路共用同一 doc_id 体系
"""
import re
from collections import Counter, defaultdict

import numpy as np
import jieba

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
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2 or not parts[1].strip():
                    continue
                tokens = tokenize(parts[1])
                doc_ids.append(parts[0])
                doc_len.append(len(tokens))
                for tok, tf in Counter(tokens).items():
                    postings[tok][0].append(len(doc_ids) - 1)
                    postings[tok][1].append(tf)
        doc_len = np.array(doc_len, dtype=np.int32)
        np_postings = {t: (np.array(li, dtype=np.int32), np.array(lt, dtype=np.float32))
                       for t, (li, lt) in postings.items()}
        if verbose:
            print("docs: {} | vocab: {} | avgdl: {:.1f}".format(
                len(doc_ids), len(np_postings), float(doc_len.mean())))
        return cls(doc_ids, doc_len, np_postings, k1=k1, b=b)

    def search(self, query, topk):
        """返回 [(doc_id, score, tokens)] 按分数降序; 查询词各计一次(标准 BM25 打分)。

        tokens 为查询的命中词(用于前端高亮), 与打分用词一致。
        """
        k1, b = self.k1, self.b
        norm = k1 * (1.0 - b + b * self.doc_len / max(self.avgdl, 1e-9))

        q_tokens = tokenize(query)
        idx_parts, score_parts, hit_tokens = [], [], []
        for tok in set(q_tokens):
            entry = self.postings.get(tok)
            if entry is None:
                continue
            idx, tf = entry
            denom = tf + norm[idx]
            idx_parts.append(idx)
            score_parts.append(self.idf[tok] * tf * (k1 + 1.0) / denom)
            hit_tokens.append(tok)
        if not idx_parts:
            return []

        all_idx = np.concatenate(idx_parts)
        all_sc = np.concatenate(score_parts)
        uniq, inv = np.unique(all_idx, return_inverse=True)
        scores = np.bincount(inv, weights=all_sc)

        k = min(topk, len(scores))
        sel = np.argpartition(-scores, k - 1)[:k]
        top = sorted(sel, key=lambda j: -scores[j])
        return [(self.doc_ids[uniq[j]], float(scores[j]), hit_tokens) for j in top]
