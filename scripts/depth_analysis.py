# -*- coding: utf-8 -*-
"""步骤 5G: 召回深度-质量分析——各路 R@k (k=10..1000) 与多路并集互补性。

对每条 dev 查询在内存中取各路 top-1000(不落盘), 统计:
  - 每路 Recall@k(BM25 词级 / 字符 bigram / ERNIE 微调双塔 / RRF 三路融合)
  - 并集覆盖率随 k 的变化(两路 BM25+ERNIE / 三路 +字符), 互补增益 = 并集 - 最优单路

回答的问题: 浅层(小 topk)与深层(大召回集)各路强弱的重组、语义路互补价值随深度的
变化、召回率随深度的饱和。口径: qrels 内全部查询(无命中按 0 计), 与九路对照表
(排除无命中查询, n=19,898)的差异 < 0.2 点。

用法:
    python scripts/depth_analysis.py                # GPU 编码查询并检索
    python scripts/depth_analysis.py --limit 200    # 快速验证
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from evaluate import load_qrels  # noqa: E402
from src.bm25 import BM25Index, _TOKENIZERS  # noqa: E402
from src.utils import read_queries, normalize  # noqa: E402

KS = [10, 20, 50, 100, 200, 500, 1000]


def encode_queries(args, queries):
    """用微调召回模型批量编码全部查询, 返回 (n, dim) 数组。"""
    import numpy as np
    import paddle
    from paddlenlp.transformers import AutoModel, AutoTokenizer
    from src.models import SemanticIndexBase

    paddle.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)
    model.set_dict(paddle.load(args.weights))
    model.eval()

    pad_id = tokenizer.pad_token_id
    embs = []
    with paddle.no_grad():
        for i in range(0, len(queries), args.batch_size):
            batch = queries[i:i + args.batch_size]
            encoded = [tokenizer(text=q["query"], max_seq_len=args.max_seq_len) for q in batch]
            L = max(len(e["input_ids"]) for e in encoded)
            ids = np.full((len(batch), L), pad_id, dtype="int64")
            types = np.zeros((len(batch), L), dtype="int64")
            for r, e in enumerate(encoded):
                ids[r, :len(e["input_ids"])] = e["input_ids"]
                types[r, :len(e["token_type_ids"])] = e["token_type_ids"]
            out = model.get_pooled_embedding(paddle.to_tensor(ids), token_type_ids=paddle.to_tensor(types))
            embs.append(out.numpy())
    return np.concatenate(embs, axis=0)


def rrf_fuse(lists, c, topn):
    scores = {}
    for lst in lists:
        for r, did in enumerate(lst, 1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (c + r)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in ranked[:topn]]


def main(args):
    t0 = time.time()
    qrels = load_qrels(args.qrels)
    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    queries = [q for q in queries if normalize(q["query"]) in qrels]
    print("qrels 查询数: {} | 本次分析查询数: {}".format(len(qrels), len(queries)))

    print("构建 BM25 词级索引 ...", flush=True)
    bm25_word = BM25Index.build(args.corpus)
    print("构建 BM25 字符索引 ... ({:.0f}s)".format(time.time() - t0), flush=True)
    bm25_char = BM25Index.build(args.corpus, tokenizer=_TOKENIZERS["char"])
    print("索引就绪 ({:.0f}s), 编码查询({}) ...".format(time.time() - t0, args.device), flush=True)

    from src.ann import load_index
    ernie_index = load_index(os.path.join(args.index_dir, "final_index.bin"),
                             dim=args.output_emb_size, ef=args.ef)
    with open(os.path.join(args.index_dir, "doc_ids.json"), "r", encoding="utf-8") as f:
        ernie_doc_ids = json.load(f)
    embs = encode_queries(args, queries)
    print("查询编码完成 ({:.0f}s)".format(time.time() - t0), flush=True)

    arms = ["BM25 词级", "字符 bigram", "ERNIE 微调", "RRF 三路(c=5)"]
    singles = arms[:3]
    hits = {a: {k: 0 for k in KS} for a in arms}
    union2 = {k: 0 for k in KS}
    union3 = {k: 0 for k in KS}
    total_pos = 0

    for qi, item in enumerate(queries):
        q = item["query"]
        positives = qrels[normalize(q)]
        total_pos += len(positives)

        lst_word = [d for d, _s, _t in bm25_word.search(q, args.topn)]
        lst_char = [d for d, _s, _t in bm25_char.search(q, args.topn)]
        rows, _dist = ernie_index.knn_query(embs[qi:qi + 1], args.topn)
        lst_ernie = [ernie_doc_ids[r] for r in rows[0]]
        fused = rrf_fuse([lst_word, lst_char, lst_ernie], args.c, args.topn)

        rank = {}
        for name, lst in zip(arms, [lst_word, lst_char, lst_ernie, fused]):
            rank[name] = {d: i for i, d in enumerate(lst, 1)}

        for did in positives:
            per_arm = {name: rank[name].get(did, args.topn + 1) for name in arms}
            for name in arms:
                for k in KS:
                    if per_arm[name] <= k:
                        hits[name][k] += 1
            u2 = min(per_arm["BM25 词级"], per_arm["ERNIE 微调"])
            u3 = min(u2, per_arm["字符 bigram"])
            for k in KS:
                if u2 <= k:
                    union2[k] += 1
                if u3 <= k:
                    union3[k] += 1

        if (qi + 1) % 2000 == 0:
            print("  {}/{} 查询, {:.0f}s".format(qi + 1, len(queries), time.time() - t0), flush=True)

    def pct(x):
        return round(100.0 * x / total_pos, 2)

    result = {"n_queries": len(queries), "total_positives": total_pos, "ks": KS,
              "recall": {a: {k: pct(hits[a][k]) for k in KS} for a in arms},
              "union_bm25_ernie": {k: pct(union2[k]) for k in KS},
              "union3": {k: pct(union3[k]) for k in KS}}
    # 最优单路 = 每个 k 上三条单路的最大值(通常是字符 bigram)
    result["best_single"] = {
        k: max(result["recall"][a][k] for a in singles) for k in KS}
    result["互补增益(并集3-最优单路)"] = {
        k: round(result["union3"][k] - result["best_single"][k], 2) for k in KS}
    result["融合截断损失(RRF3-并集3)"] = {
        k: round(result["recall"]["RRF 三路(c=5)"][k] - result["union3"][k], 2) for k in KS}

    print()
    header = "| 召回路 | " + " | ".join("R@{}".format(k) for k in KS) + " |"
    print(header)
    print("|" + "---|" * (len(KS) + 1))
    for a in arms:
        print("| {} | ".format(a) + " | ".join(str(result["recall"][a][k]) for k in KS) + " |")
    print("| 最优单路 | " + " | ".join(str(result["best_single"][k]) for k in KS) + " |")
    print("| 并集 BM25+ERNIE | " + " | ".join(str(result["union_bm25_ernie"][k]) for k in KS) + " |")
    print("| 三路并集 | " + " | ".join(str(result["union3"][k]) for k in KS) + " |")
    print("| 互补增益(并集-最优) | " + " | ".join(
        "+{:.2f}".format(result["互补增益(并集3-最优单路)"][k]) for k in KS) + " |")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("written:", args.out)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"))
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--qrels", default=os.path.join(ROOT, "data", "qrels_dev.tsv"))
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index_finetuned_clean"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ef", type=int, default=1100, help="需 >= topn, 保证 k=1000 可用")
    ap.add_argument("--c", type=float, default=5.0)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "depth_analysis.json"))
    main(ap.parse_args())
