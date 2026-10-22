# -*- coding: utf-8 -*-
"""步骤 4/8: 对语料全量编码并构建 HNSW 向量索引。

索引保存为两个文件:
    --index-dir/final_index.bin  hnswlib 索引(行顺序 = 语料顺序)
    --index-dir/doc_ids.json     每行对应的 doc_id 列表(侧车文件, 召回时使用)

推荐使用清洗后的语料(data/corpus_clean.tsv, 29.2 万篇)而非原始 corpus.csv,
避免重复文献占据索引与候选名额。

用法:
    python scripts/build_index.py
    python scripts/build_index.py --weights weights/recall/model_180 --device gpu
"""
import argparse
import json
import os
import sys
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

from src.models import SemanticIndexBase
from src.data import convert_bi_example, make_batchify, create_dataloader
from src.ann import build_index, save_index
from src.utils import load_corpus


def main(args):
    paddle.set_device(args.device)
    doc_ids, texts = load_corpus(args.corpus)
    print("语料文献数:", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)

    if args.weights:
        state = paddle.load(args.weights)
        model.set_dict(state)
        print("loaded weights:", args.weights)

    corpus_ds = MapDataset([{"text": t} for t in texts])
    trans_fn = partial(convert_bi_example, tokenizer=tokenizer, max_seq_length=args.max_seq_len)
    loader = create_dataloader(
        corpus_ds, mode="infer", batch_size=args.batch_size,
        batchify_fn=make_batchify("bi_infer", tokenizer), trans_fn=trans_fn)

    index, _ = build_index(
        dim=args.output_emb_size, data_loader=loader, model=model,
        ef_construction=args.ef_construction, m=args.m, max_elements=args.max_elements)

    os.makedirs(args.index_dir, exist_ok=True)
    index_path = os.path.join(args.index_dir, "final_index.bin")
    save_index(index, index_path)
    with open(os.path.join(args.index_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print("written:", index_path)
    print("written:", os.path.join(args.index_dir, "doc_ids.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"),
                    help="推荐清洗语料; 也可用原始 datasets/recall/corpus.csv")
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"),
                    help="微调后的模型参数; 传空字符串则用预训练底座")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--max-elements", type=int, default=1000000)
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index"))
    args = ap.parse_args()
    if args.weights == "":
        args.weights = None
    main(args)
