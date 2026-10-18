# -*- coding: utf-8 -*-
"""步骤 3/8: 微调 ERNIE-1.0 双塔召回模型(In-batch Negatives 对比学习)。

训练目标: 查询与正例文献的余弦相似度高于批内其他文献。
每 save_steps 步保存一次 checkpoint 到 --save-dir/model_<step>/。

用法(CPU 训练, 约 4000 对、3 个 epoch):
    python scripts/train_recall.py
    python scripts/train_recall.py --device gpu --epochs 3 --save-steps 100
"""
import argparse
import os
import sys
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup, ErnieTokenizer, ErnieModel

from src.models import SemanticIndexBatchNeg
from src.data import read_text_pair, convert_bi_example, make_batchify, create_dataloader


def main(args):
    paddle.set_device(args.device)
    # 固定随机种子(开题报告要求记录), 保证训练可复现
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    print("seed:", args.seed)
    tokenizer = ErnieTokenizer.from_pretrained(args.base_model)
    pretrained = ErnieModel.from_pretrained(args.base_model)

    train_ds = MapDataset(list(read_text_pair(args.train)))
    print("训练配对数:", len(train_ds))

    trans_fn = partial(convert_bi_example, tokenizer=tokenizer, max_seq_length=args.max_seq_len)
    train_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size,
        batchify_fn=make_batchify("bi", tokenizer), trans_fn=trans_fn)

    model = SemanticIndexBatchNeg(
        pretrained, margin=args.margin, scale=args.scale, output_emb_size=args.output_emb_size)

    epochs = args.epochs
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(args.lr, num_training_steps, args.warmup_proportion)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params)

    os.makedirs(args.save_dir, exist_ok=True)
    global_step, tic = 0, 0.0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            query_input_ids, query_types, title_input_ids, title_types = batch
            loss = model(
                query_input_ids=query_input_ids, title_input_ids=title_input_ids,
                query_token_type_ids=query_types, title_token_type_ids=title_types)

            global_step += 1
            if global_step % args.log_steps == 0:
                speed = args.log_steps / max(1e-6, _now() - tic)
                print("epoch {} step {} loss {:.5f} ({:.2f} step/s)".format(epoch, global_step, float(loss), speed))
                tic = _now()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.save_steps == 0:
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
    ap.add_argument("--train", default=os.path.join(ROOT, "datasets", "recall", "train.csv"))
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-proportion", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "weights", "recall"))
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="随机种子, 训练日志会记录")
    main(ap.parse_args())
