# -*- coding: utf-8 -*-
"""步骤 2/8: 构建查询-相关文献映射(qrels), 支持多正例。

把 dev/test 配对文件中的正例文献文本映射为 doc_id, 产出统一评估所需的 qrels。
正例未命中语料的查询会被排除并记录(可评价查询数在评估时必须报告)。

用法:
    python scripts/build_qrels.py                     # 默认 dev + test
    python scripts/build_qrels.py --splits dev        # 只处理 dev
"""
import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import normalize


def load_corpus_map(clean_path):
    norm2id = {}
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                norm2id[normalize(parts[1])] = parts[0]
    return norm2id


def main(args):
    clean_path = os.path.join(args.data_dir, "corpus_clean.tsv")
    norm2id = load_corpus_map(clean_path)
    os.makedirs(args.data_dir, exist_ok=True)

    for split in args.splits:
        pair_path = os.path.join(args.datasets_dir, "{}.csv".format(split))
        qrels = defaultdict(set)      # 规范化查询 -> 正例 doc_id 集合
        display = {}                  # 规范化查询 -> 原始查询文本(首次出现)
        excluded, n_lines = [], 0

        with open(pair_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    continue
                n_lines += 1
                q, pos = parts[0], parts[1]
                display.setdefault(normalize(q), q)
                did = norm2id.get(normalize(pos))
                if did is None:
                    excluded.append((q, pos))
                else:
                    qrels[normalize(q)].add(did)

        qrels_path = os.path.join(args.data_dir, "qrels_{}.tsv".format(split))
        with open(qrels_path, "w", encoding="utf-8") as f:
            for nq in sorted(qrels):
                for did in sorted(qrels[nq]):
                    f.write("{}\t{}\n".format(display[nq], did))

        with open(os.path.join(args.data_dir, "excluded_{}.txt".format(split)), "w", encoding="utf-8") as f:
            for q, pos in excluded:
                f.write("{}\t{}\n".format(q, pos))

        multi = sum(1 for s in qrels.values() if len(s) > 1)
        print("=== {} ===".format(split))
        print("配对行数: {} | 唯一查询: {} | 可评价: {} | 排除正例: {} | 多正例查询: {}".format(
            n_lines, len(display), len(qrels), len(excluded), multi))
        if len(display) - len(qrels) > 0:
            print("警告: {} 个查询的所有正例均未命中语料, 已排除(详见 excluded_{}.txt)".format(
                len(display) - len(qrels), split))
        print("written:", qrels_path)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--datasets-dir", default=os.path.join(ROOT, "datasets", "recall"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    main(ap.parse_args())
