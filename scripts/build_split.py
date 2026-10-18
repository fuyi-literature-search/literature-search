# -*- coding: utf-8 -*-
"""步骤 2B/8: 查询级分组划分(防训练/评估泄漏)。

开题报告要求: "召回与排序数据共同按查询及近重复查询分组划分; 测试查询不参与训练、
负例挖掘或调参"。本脚本把全部数据(召回 train/dev/test + 精排 train/dev/test)的查询
做近重复聚类, 凡与 dev/test 查询同组的训练查询一律从训练集中剔除, 输出清洗后的
训练集与统计报告。

近重复规则(两条满足其一即同组, Union-Find 合并):
  1. 规范化文本完全相同(normalize: 去全部空白)
  2. 字符 3-gram Jaccard >= 阈值(默认 0.7)
     候选对由 MinHash(32 个哈希) + LSH(8 band x 4 row) 生成后精确验证

用法:
    python scripts/build_split.py                   # 默认阈值 0.7
    python scripts/build_split.py --threshold 0.6
"""
import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from src.utils import normalize

N_HASH = 32
N_BAND = 8
BAND_ROWS = N_HASH // N_BAND          # 4
PRIME = (1 << 31) - 1
MAX_BUCKET = 100                      # LSH 桶过大视为公共查询聚集, 跳过(防 O(n^2))


def grams(text):
    """字符 3-gram 集合; 短文本(去掉首尾空白后)返回 None, 只按精确匹配分组。"""
    if len(text) < 3:
        return None
    return set(text[i:i + 3] for i in range(len(text) - 2))


def collect_queries(args):
    """扫描全部数据文件, 返回 [(norm_query, role)], 同一查询可多角色。"""
    sources = [
        (os.path.join(args.datasets_dir, "recall", "train.csv"), "recall_train", 1, 0),
        (os.path.join(args.datasets_dir, "recall", "dev.csv"), "recall_dev", 1, 0),
        (os.path.join(args.datasets_dir, "recall", "test.csv"), "recall_test", 1, 0),
        (os.path.join(args.datasets_dir, "rank", "train_pairwise.csv"), "rank_train", 1, 1),
        (os.path.join(args.datasets_dir, "rank", "dev_pairwise.csv"), "rank_dev", 1, 1),
        (os.path.join(args.datasets_dir, "rank", "test_pairwise.csv"), "rank_test", 1, 1),
    ]
    pairs = []
    for path, role, q_col, skip_header in sources:
        if not os.path.isfile(path):
            print("跳过缺失文件:", path)
            continue
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if skip_header and i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= q_col or not parts[q_col].strip():
                    continue
                pairs.append((normalize(parts[q_col]), role))
                n += 1
        print("{:<24} 行数 {:>8}".format(role, n))
    return pairs


class UnionFind(object):
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def minhash_signatures(queries):
    """对唯一查询计算 MinHash 签名(numpy 向量化)。返回 (n, N_HASH) uint64。"""
    vocab = {}
    grams_per_q = []
    flat_ids = []
    for q in queries:
        g = grams(q)
        ids = []
        if g:
            for x in g:
                vid = vocab.get(x)
                if vid is None:
                    vid = len(vocab)
                    vocab[x] = vid
                ids.append(vid)
        grams_per_q.append(ids)
        flat_ids.extend(ids)

    rng = np.random.RandomState(42)
    a = rng.randint(1, PRIME, size=N_HASH, dtype=np.int64)
    b = rng.randint(0, PRIME, size=N_HASH, dtype=np.int64)
    flat = np.array(flat_ids, dtype=np.int64)
    offsets = np.cumsum([0] + [len(g) for g in grams_per_q])
    empty = offsets[1:] == offsets[:-1]

    n = len(queries)
    sig = np.full((n, N_HASH), np.iinfo(np.int64).max, dtype=np.int64)
    for k in range(N_HASH):                     # 逐列计算, 避免物化 (总gram数, 32) 大矩阵
        col = (flat * int(a[k]) + int(b[k])) % PRIME
        col_min = np.minimum.reduceat(col, offsets[:-1])
        col_min = np.where(empty, np.iinfo(np.int64).max, col_min)
        sig[:, k] = np.minimum(sig[:, k], col_min)
    print("唯一查询: {} | 3-gram 词表: {}".format(n, len(vocab)))
    return sig


def near_dup_groups(queries, threshold):
    """返回 UnionFind 分组后的 parent 数组。"""
    n = len(queries)
    uf = UnionFind(n)

    # (1) 精确同文本: 建索引时已唯一化, 无需处理(调用方传入唯一查询)

    # (2) MinHash + LSH 候选, 精确 Jaccard 验证
    sig = minhash_signatures(queries)
    gram_sets = [grams(q) or set() for q in queries]
    for band in range(N_BAND):
        cols = sig[:, band * BAND_ROWS:(band + 1) * BAND_ROWS]
        _, inverse, counts = np.unique(cols, axis=0, return_inverse=True, return_counts=True)
        buckets = defaultdict(list)
        for qi, bi in enumerate(inverse):
            if counts[bi] > 1:
                buckets[bi].append(qi)
        n_cand = 0
        for members in buckets.values():
            if len(members) > MAX_BUCKET:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    gi, gj = gram_sets[i], gram_sets[j]
                    if not gi or not gj:
                        continue
                    inter = len(gi & gj)
                    if inter == 0:
                        continue
                    jac = inter / float(len(gi) + len(gj) - inter)
                    if jac >= threshold:
                        uf.union(i, j)
                        n_cand += 1
        print("band {}/{} 完成".format(band + 1, N_BAND))
    for i in range(n):                          # 路径压缩, 使 parent[i] 成为组代表元
        uf.parent[i] = uf.find(i)
    return uf.parent


def main(args):
    print("=== 1. 扫描各数据集查询 ===")
    pairs = collect_queries(args)

    print("=== 2. 唯一化 + 近重复分组 ===")
    uniq = sorted(set(q for q, _ in pairs))
    index = {q: i for i, q in enumerate(uniq)}
    parent = near_dup_groups(uniq, args.threshold)

    # 组内角色聚合
    group_roles = defaultdict(set)
    for q, role in pairs:
        group_roles[index[q]].add(role)          # 用代表元聚合
    root_roles = defaultdict(set)
    for qi in range(len(uniq)):
        root_roles[parent[qi]] |= group_roles[qi]

    heldout = {"recall_dev", "recall_test", "rank_dev", "rank_test"}
    contaminated_roots = {r for r, roles in root_roles.items() if roles & heldout}
    is_excluded = [parent[i] in contaminated_roots for i in range(len(uniq))]
    print("唯一查询: {} | 组数: {} | 含 dev/test 的组: {} | 需剔除查询: {}".format(
        len(uniq), len(root_roles), len(contaminated_roots), sum(is_excluded)))

    print("=== 3. 过滤训练集并落盘 ===")
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    jobs = [
        (os.path.join(args.datasets_dir, "recall", "train.csv"),
         "recall_train", data_dir + "/recall_train_clean.tsv", 1, 0),
        (os.path.join(args.datasets_dir, "rank", "train_pairwise.csv"),
         "rank_train", data_dir + "/rank_train_clean.tsv", 1, 1),
    ]
    stats = {}
    for path, role, out_path, q_col, skip_header in jobs:
        if not os.path.isfile(path):
            continue
        kept = excluded = 0
        with open(path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
            for i, line in enumerate(f):
                if skip_header and i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= q_col or not parts[q_col].strip():
                    continue
                if is_excluded[index[normalize(parts[q_col])]]:
                    excluded += 1
                else:
                    out.write("\t".join(parts) + "\n")
                    kept += 1
        stats[role] = (kept, excluded)
        print("{:<24} 保留 {:>8} | 剔除 {:>8} ({:.1%})".format(
            role, kept, excluded, excluded / max(1, kept + excluded)))
        print("written:", out_path)

    with open(os.path.join(data_dir, "split_stats.txt"), "w", encoding="utf-8") as f:
        f.write("查询分组划分报告 (3-gram Jaccard >= {})\n".format(args.threshold))
        f.write("唯一查询: {} | 组数: {} | 剔除查询: {}\n".format(
            len(uniq), len(root_roles), sum(is_excluded)))
        for role, (kept, excluded) in stats.items():
            f.write("{}: 保留 {} | 剔除 {}\n".format(role, kept, excluded))
    print("written:", os.path.join(data_dir, "split_stats.txt"))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--threshold", type=float, default=0.7, help="近重复 Jaccard 阈值")
    main(ap.parse_args())
