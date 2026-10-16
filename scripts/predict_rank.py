# -*- coding: utf-8 -*-
"""步骤 7/8: 用训练好的精排模型对测试集句对打分排序。

用法:
    python scripts/predict_rank.py
    python scripts/predict_rank.py --weights weights/rank/model_400/model_state.pdparams
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

from src.models import PairwiseMatching
from src.data import read_text_pair, convert_pairwise_example, make_batchify, create_dataloader


@paddle.no_grad()
def predict(model, data_loader):
    model.eval()
    probs = []
    for input_ids, token_type_ids in data_loader:
        prob = model.predict(input_ids=input_ids, token_type_ids=token_type_ids)
        probs.append(prob.numpy())
    return np.concatenate(probs, axis=0)


def main(args):
    paddle.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = PairwiseMatching(pretrained)
    model.set_dict(paddle.load(args.weights))
    print("loaded weights:", args.weights)

    pairs = list(read_text_pair(args.test))
    test_ds = MapDataset(pairs)
    trans_fn = partial(convert_pairwise_example, tokenizer=tokenizer,
                       max_seq_length=args.max_seq_len, phase="predict")
    loader = create_dataloader(
        test_ds, mode="infer", batch_size=args.batch_size,
        batchify_fn=make_batchify("pair_predict", tokenizer), trans_fn=trans_fn)

    probs = predict(model, loader)

    results = []
    for pair, prob in zip(pairs, probs):
        results.append({"query": pair["query"], "title": pair["title"], "prob": float(prob[0])})
    results.sort(key=lambda x: x["prob"], reverse=True)

    for r in results[: args.show_top]:
        print("{:.4f}\t{}\t{}".format(r["prob"], r["query"][:30], r["title"][:50]))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write("{}\t{}\t{:.6f}\n".format(r["query"], r["title"], r["prob"]))
        print("written:", args.output)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", default=os.path.join(ROOT, "datasets", "rank", "test_pairwise.csv"))
    ap.add_argument("--base-model", default="ernie-3.0-medium-zh")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "rank", "model_state.pdparams"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--max-seq-len", type=int, default=128,
                    help="必须与训练一致")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--show-top", type=int, default=10)
    ap.add_argument("--output", default="")
    main(ap.parse_args())
