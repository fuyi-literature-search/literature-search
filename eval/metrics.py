# -*- coding: utf-8 -*-
"""排序效果指标。全部按查询计算后取宏平均。

约定: ranked_ids 为按分数降序排列的 doc_id 列表; rel_set 为相关文献 doc_id 集合。

指标口径:
  recall@k   = |top-k ∩ rel| / |rel|           (多正例时为比例, 单正例时退化为 0/1)
  hit@k      = 1 若 top-k 中含至少一个相关文献
  mrr@k      = 1/rank(第一个相关文献), rank<=k, 否则 0
  ndcg@k     = DCG@k / IDCG@k, 二元相关增益
"""
import math


def recall_at_k(ranked_ids, rel_set, k):
    if not rel_set:
        return 0.0
    top = ranked_ids[:k]
    return sum(1 for d in top if d in rel_set) / float(len(rel_set))


def hit_at_k(ranked_ids, rel_set, k):
    return 1.0 if any(d in rel_set for d in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids, rel_set, k):
    for i, d in enumerate(ranked_ids[:k]):
        if d in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids, rel_set, k):
    if not rel_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked_ids[:k])
        if d in rel_set
    )
    ideal = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(run_queries, qrels, ks=(50, 100, 200), mrr_k=10, hit_k=10, ndcg_k=10):
    """run_queries: {norm_query: [doc_id 按分数降序]}; qrels: {norm_query: set(doc_id)}

    返回 (aggregate dict, per_query dict)。只统计两边都存在的查询。
    """
    per_query = {}
    common = [q for q in qrels if q in run_queries]
    for q in common:
        ranked = run_queries[q]
        rel = qrels[q]
        entry = {"rel": len(rel), "ranked": len(ranked)}
        for k in ks:
            entry["recall@{}".format(k)] = recall_at_k(ranked, rel, k)
        entry["hit@{}".format(hit_k)] = hit_at_k(ranked, rel, hit_k)
        entry["mrr@{}".format(mrr_k)] = mrr_at_k(ranked, rel, mrr_k)
        entry["ndcg@{}".format(ndcg_k)] = ndcg_at_k(ranked, rel, ndcg_k)
        per_query[q] = entry

    n = len(common)
    if n == 0:
        return {}, per_query
    agg = {"n_queries": n}
    for name in next(iter(per_query.values())):
        if name in ("rel", "ranked"):
            continue
        agg[name] = sum(e[name] for e in per_query.values()) / n
    return agg, per_query


def paired_bootstrap(per_query_a, per_query_b, metric, n_boot=1000, seed=42):
    """两个 run 的查询级配对差值 + bootstrap 95% 置信区间。

    返回 (mean_diff, ci_low, ci_high, n)。
    """
    import random

    queries = [q for q in per_query_a if q in per_query_b]
    if not queries:
        return 0.0, 0.0, 0.0, 0
    da = [per_query_a[q][metric] for q in queries]
    db = [per_query_b[q][metric] for q in queries]
    diffs = [a - b for a, b in zip(da, db)]
    mean_diff = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return mean_diff, lo, hi, n


if __name__ == "__main__":
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert abs(recall_at_k(ranked, rel, 3) - 0.5) < 1e-9
    assert hit_at_k(ranked, rel, 2) == 1.0
    assert abs(mrr_at_k(ranked, rel, 10) - 0.5) < 1e-9
    assert abs(ndcg_at_k(["a", "b"], {"a"}, 2) - 1.0) < 1e-9
    assert abs(ndcg_at_k(["b", "a"], {"a"}, 2) - 0.6309297535714575) < 1e-9
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    print("metrics self-test passed")
