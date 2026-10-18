# -*- coding: utf-8 -*-
"""共享工具: 文本规范化与稳定 doc_id。

规范化规则: 删除全部空白字符(中文文本中空白无区分意义), 保留大小写。
doc_id 规则: sha1(规范化文本) 前 16 位十六进制。
构建索引 / 生成 qrels / 评估的所有环节必须使用同一规范化, 保证 ID 一致。
"""
import hashlib
import re

_WS_RE = re.compile(r"\s+")


def normalize(text):
    return _WS_RE.sub("", text)


def doc_id_of(norm_text):
    return hashlib.sha1(norm_text.encode("utf-8")).hexdigest()[:16]


def read_queries(data_path):
    """批量召回: 取每行第一列作为查询(去重且保持原顺序)。"""
    queries = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.rstrip("\n").split("\t")[0]
            if q.strip():
                queries.append({"query": q})
    seen = set()
    unique = []
    for item in queries:
        if item["query"] not in seen:
            seen.add(item["query"])
            unique.append(item)
    return unique


def load_corpus(path):
    """加载语料, 返回 (doc_ids, texts)。

    支持两种格式:
      - corpus.csv            : 每行一篇文献(标题+关键词连写), doc_id 按行号生成 "line{i}"
      - data/corpus_clean.tsv : doc_id \t 文本, 使用文件给定的稳定 doc_id
    """
    doc_ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and len(parts[0]) == 16:
                doc_ids.append(parts[0])
                texts.append(parts[1])
            else:
                doc_ids.append("line{}".format(i))
                texts.append(line)
    return doc_ids, texts
