# -*- coding: utf-8 -*-
"""步骤 8/8: 端到端交互式搜索 - 召回 Top-50 候选, 交叉编码器精排后展示。

流程: 输入查询 -> ERNIE 双塔编码 -> HNSW 检索 Top-K 候选 -> 交叉编码器逐对打分
      -> 按相关性概率降序输出。

用法:
    python scripts/search.py --query "热处理对尼龙6结晶的影响"
    python scripts/search.py                       # 交互式输入
"""
import argparse
import os
import sys
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

from src.models import SemanticIndexBase, PairwiseMatching
from src.data import convert_bi_example, convert_pairwise_example, make_batchify, create_dataloader
from src.ann import load_index
from src.utils import load_corpus


@paddle.no_grad()
def rank_predict(model, data_loader):
    model.eval()
    probs = []
    for input_ids, token_type_ids in data_loader:
        probs.append(model.predict(input_ids=input_ids, token_type_ids=token_type_ids).numpy())
    return np.concatenate(probs, axis=0)


def main(args):
    paddle.set_device(args.device)

    # ---------- 召回准备 ----------
    doc_ids, doc_texts = load_corpus(args.corpus)
    index = load_index(os.path.join(args.index_dir, "final_index.bin"),
                       dim=args.output_emb_size, ef=args.ef)
    recall_tokenizer = AutoTokenizer.from_pretrained(args.recall_model)
    recall_pretrained = AutoModel.from_pretrained(args.recall_model)
    recall_model = SemanticIndexBase(recall_pretrained, output_emb_size=args.output_emb_size)
    recall_model.set_dict(paddle.load(args.recall_weights))
    print("recall model loaded:", args.recall_weights)

    # ---------- 精排准备(tokenizer 与精排模型必须配套, 输入长度与训练一致) ----------
    rank_tokenizer = AutoTokenizer.from_pretrained(args.rank_model)
    rank_pretrained = AutoModel.from_pretrained(args.rank_model)
    rank_model = PairwiseMatching(rank_pretrained)
    rank_model.set_dict(paddle.load(args.rank_weights))
    print("rank model loaded:", args.rank_weights)

    rank_trans = partial(convert_pairwise_example, tokenizer=rank_tokenizer,
                         max_seq_length=args.rank_max_seq_len, phase="predict")

    recall_trans = partial(convert_bi_example, tokenizer=recall_tokenizer, max_seq_length=args.recall_max_seq_len)

    # ---------- 交互循环 ----------
    while True:
        try:
            query = args.query or input("请输入需要查询的文献 (直接回车退出): ").strip()
        except EOFError:
            break
        if not query:
            break

        query_ds = MapDataset([{"query": query}])
        query_loader = create_dataloader(
            query_ds, mode="infer", batch_size=1,
            batchify_fn=make_batchify("bi_infer", recall_tokenizer), trans_fn=recall_trans)
        query_embedding = next(iter(recall_model.get_semantic_embedding(query_loader)))

        recalled_idx, distances = index.knn_query(query_embedding.numpy(), args.topk)
        recall_data = []
        for idx, doc_row in enumerate(recalled_idx[0]):
            score = 1.0 - float(distances[0][idx])
            did = doc_ids[doc_row] if doc_ids else "line{}".format(doc_row)
            recall_data.append({"query": query, "title": doc_texts[doc_row], "doc_id": did, "recall_score": score})

        print("\n[召回] Top-{} 候选(前5条预览):".format(args.topk))
        for item in recall_data[:5]:
            print("  {:.4f}  {}".format(item["recall_score"], item["title"][:60]))

        # ---------- 精排 ----------
        rank_items = [{"query": d["query"], "title": d["title"]} for d in recall_data]
        rank_ds = MapDataset(rank_items)
        rank_loader = create_dataloader(
            rank_ds, mode="infer", batch_size=16,
            batchify_fn=make_batchify("pair_predict", rank_tokenizer), trans_fn=rank_trans)
        probs = rank_predict(rank_model, rank_loader)

        for item, prob in zip(rank_items, probs):
            item["rank_prob"] = float(prob[0])
        rank_items.sort(key=lambda x: x["rank_prob"], reverse=True)

        print("\n[精排] Top-{} 结果:".format(args.show_top))
        for i, item in enumerate(rank_items[: args.show_top], start=1):
            print("  {:>2}. P={:.4f}  {}".format(i, item["rank_prob"], item["title"][:70]))
        print()

        if args.query:
            break


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default="", help="查询文本; 留空进入交互模式")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"),
                    help="必须与建索引时使用的语料一致")
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index"))
    ap.add_argument("--recall-model", default="ernie-1.0")
    ap.add_argument("--recall-weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--rank-model", default="ernie-3.0-medium-zh")
    ap.add_argument("--rank-weights", default=os.path.join(ROOT, "weights", "rank", "rank_model_state.pdparams"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--ef", type=int, default=100)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--recall-max-seq-len", type=int, default=60)
    ap.add_argument("--rank-max-seq-len", type=int, default=128)
    ap.add_argument("--show-top", type=int, default=10)
    main(ap.parse_args())
