# -*- coding: utf-8 -*-
"""步骤 6/8: 训练交叉编码器精排模型(pairwise margin ranking loss)。

每 eval_steps 步在验证集上计算 AUC 并保存 checkpoint。

用法:
    python scripts/train_rank.py
    python scripts/train_rank.py --device gpu --save-steps 400
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
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup

from src.models import PairwiseMatching
from src.data import (read_rank_triplet, read_rank_labeled, convert_pairwise_example,
                      make_batchify, create_dataloader)


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        pos_probs = model.predict(input_ids=input_ids, token_type_ids=token_type_ids)
        preds = np.concatenate((1.0 - pos_probs, pos_probs), axis=1)
        metric.update(preds=preds, labels=labels)
        auc = metric.accumulate()
    print("dev AUC: {:.3}".format(auc))
    metric.reset()
    model.train()
    return auc


def main(args):
    paddle.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)

    train_ds = MapDataset(list(read_rank_triplet(args.train)))
    dev_ds = MapDataset(list(read_rank_labeled(args.dev)))
    print("训练三元组: {} | 验证对: {}".format(len(train_ds), len(dev_ds)))

    trans_train = partial(convert_pairwise_example, tokenizer=tokenizer,
                          max_seq_length=args.max_seq_len, phase="train")
    trans_eval = partial(convert_pairwise_example, tokenizer=tokenizer,
                         max_seq_length=args.max_seq_len, phase="eval")
    train_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size,
        batchify_fn=make_batchify("pair_train", tokenizer), trans_fn=trans_train)
    dev_loader = create_dataloader(
        dev_ds, mode="infer", batch_size=args.batch_size,
        batchify_fn=make_batchify("pair_eval", tokenizer), trans_fn=trans_eval)

    model = PairwiseMatching(pretrained, margin=args.margin)

    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = LinearDecayWithWarmup(args.lr, num_training_steps, args.warmup_proportion)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params)
    metric = paddle.metric.Auc()

    os.makedirs(args.save_dir, exist_ok=True)
    global_step, tic = 0, 0.0
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            pos_ids, pos_types, neg_ids, neg_types = batch
            loss = model(pos_input_ids=pos_ids, neg_input_ids=neg_ids,
                         pos_token_type_ids=pos_types, neg_token_type_ids=neg_types)

            global_step += 1
            if global_step % args.log_steps == 0:
                speed = args.log_steps / max(1e-6, _now() - tic)
                print("epoch {} step {} loss {:.5f} ({:.2f} step/s)".format(epoch, global_step, float(loss), speed))
                tic = _now()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_steps == 0:
                evaluate(model, metric, dev_loader)
                ckpt = os.path.join(args.save_dir, "model_{}".format(global_step))
                os.makedirs(ckpt, exist_ok=True)
                paddle.save(model.state_dict(), os.path.join(ckpt, "model_state.pdparams"))
                tokenizer.save_pretrained(ckpt)
                print("saved:", ckpt)

    print("done. checkpoints in", args.save_dir)


def _now():
    import time
    return time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=os.path.join(ROOT, "datasets", "rank", "train_pairwise.csv"))
    ap.add_argument("--dev", default=os.path.join(ROOT, "datasets", "rank", "dev_pairwise.csv"))
    ap.add_argument("--base-model", default="ernie-3.0-medium-zh")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-proportion", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128,
                    help="训练与推理必须一致")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "weights", "rank"))
    main(ap.parse_args())
