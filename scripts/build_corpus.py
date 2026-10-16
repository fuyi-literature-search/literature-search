# -*- coding: utf-8 -*-
"""步骤 1/8: 语料清洗与稳定 doc_id。

读取原始语料(每行一篇文献), 去除空白差异造成的重复, 为每篇生成稳定 doc_id
(sha1 前 16 位)。后续 BM25 索引、HNSW 索引、qrels 全部使用同一套 doc_id。

用法:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --corpus datasets/recall/corpus.csv --out-dir data
"""
import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import normalize, doc_id_of


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    norm2id = {}
    id2text = {}
    stats = Counter()

    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            if not text.strip():
                stats["blank"] += 1
                continue
            stats["total"] += 1
            norm = normalize(text)
            if norm in norm2id:
                stats["dup"] += 1
                continue
            did = doc_id_of(norm)
            norm2id[norm] = did
            id2text[did] = text
            stats["unique"] += 1

    clean_path = os.path.join(args.out_dir, "corpus_clean.tsv")
    with open(clean_path, "w", encoding="utf-8") as f:
        for did, text in id2text.items():
            f.write("{}\t{}\n".format(did, text))

    print("输入总行数: {} | 空白行: {} | 重复: {} | 清洗后: {}".format(
        stats["total"], stats["blank"], stats["dup"], stats["unique"]))
    print("written:", clean_path)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "datasets", "recall", "corpus.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    main(ap.parse_args())
