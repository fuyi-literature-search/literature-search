# -*- coding: utf-8 -*-
"""数据读取与批处理整理。

读取器(生成器, 逐行产出 dict):
    read_text_pair       : 查询 \t 正例文献            (召回训练对 / 精排测试对)
    read_rank_triplet    : 查询 \t 正例 \t 负例        (精排训练三元组, 含表头)
    read_rank_labeled    : 查询 \t 文献 \t 标签        (精排验证, 含表头)
    read_queries         : 查询文件的第一列            (批量召回)

转换器(配合 tokenizer 把 dict 变成 id 序列):
    convert_bi_example       : 双塔单文本 (query 或 title 各自编码)
    convert_pairwise_example : 交叉编码器句对, phase = train / eval / predict

整理器(collate_fn, 把一个 batch 的变长序列 pad 成矩阵):
    batchify_bi  / batchify_bi_eval
    batchify_pairwise_train / batchify_pairwise_eval
"""
import paddle
from paddlenlp.data import Pad, Stack, Tuple


# ---------------------------------------------------------------- 读取器
def read_text_pair(data_path):
    """查询 \t 正例文献; 兼容行尾多出的标签列。"""
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            yield {"query": parts[0], "title": parts[1]}


def read_rank_triplet(data_path):
    """精排训练三元组, 首行为表头: query \t title(正例) \t neg_title(负例)。"""
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # 跳过表头
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            yield {"query": parts[0], "title": parts[1], "neg_title": parts[2]}


def read_rank_labeled(data_path):
    """精排验证对, 首行为表头: query \t title \t label。"""
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            yield {"query": parts[0], "title": parts[1], "label": parts[2]}


def read_queries(data_path):
    """批量召回: 取每行第一列作为查询。"""
    queries = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.rstrip("\n").split("\t")[0]
            if q.strip():
                queries.append({"query": q})
    # 去重且保持原顺序(重复查询只需召回一次)
    seen = set()
    unique = []
    for item in queries:
        if item["query"] not in seen:
            seen.add(item["query"])
            unique.append(item)
    return unique


# ---------------------------------------------------------------- 转换器
def convert_bi_example(example, tokenizer, max_seq_length=60):
    """双塔: example 中每个字段独立编码, 返回拼接的 id 序列列表。"""
    result = []
    for _, text in example.items():
        encoded = tokenizer(text=text, max_seq_len=max_seq_length)
        result += [encoded["input_ids"], encoded["token_type_ids"]]
    return result


def convert_pairwise_example(example, tokenizer, max_seq_length=128, phase="train"):
    """交叉编码器句对编码。

    train  : (query, 正例) 与 (query, 负例) 分别编码, 返回 4 条序列
    eval   : (query, title) 编码 + label
    predict: 仅 (query, title) 编码
    """
    if phase == "train":
        pos_inputs = tokenizer(text=example["query"], text_pair=example["title"], max_seq_len=max_seq_length)
        neg_inputs = tokenizer(text=example["query"], text_pair=example["neg_title"], max_seq_len=max_seq_length)
        return [
            pos_inputs["input_ids"], pos_inputs["token_type_ids"],
            neg_inputs["input_ids"], neg_inputs["token_type_ids"],
        ]

    inputs = tokenizer(text=example["query"], text_pair=example["title"], max_seq_len=max_seq_length)
    if phase == "eval":
        return inputs["input_ids"], inputs["token_type_ids"], example["label"]
    if phase == "predict":
        return inputs["input_ids"], inputs["token_type_ids"]
    raise ValueError("not supported phase: {}".format(phase))


# ---------------------------------------------------------------- 整理器
def _make_pair_batchify(tokenizer):
    """双塔: 每样本 [query_ids, query_types, title_ids, title_types] 各自 pad。"""

    def batchify_fn(samples):
        fn = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
        )
        return fn(samples)

    return batchify_fn


def _make_single_pair_batchify(tokenizer, with_label):
    """交叉编码器 eval/predict: 单句对 pad; eval 额外堆叠 label。"""

    def batchify_fn(samples):
        if with_label:
            fn = Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
                Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
                Stack(dtype="int64"),
            )
        else:
            fn = Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
                Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
            )
        return fn(samples)

    return batchify_fn


def make_batchify(kind, tokenizer):
    """按任务类型构造 collate_fn。

    kind:
      bi            - 双塔训练(4 列 pad)
      bi_infer      - 双塔推理(2 列 pad, query 或 title 单独编码)
      pair_train    - 交叉编码器训练(4 列 pad)
      pair_eval     - 交叉编码器验证(2 列 pad + label)
      pair_predict  - 交叉编码器预测(2 列 pad)
    """
    if kind == "bi":
        return _make_pair_batchify(tokenizer)
    if kind == "bi_infer":
        def batchify_fn(samples):
            fn = Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
                Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
            )
            return fn(samples)
        return batchify_fn
    if kind == "pair_train":
        return _make_pair_batchify(tokenizer)
    if kind == "pair_eval":
        return _make_single_pair_batchify(tokenizer, with_label=True)
    if kind == "pair_predict":
        return _make_single_pair_batchify(tokenizer, with_label=False)
    raise ValueError("unknown batchify kind: {}".format(kind))


def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    """map 转换 + 批采样 + collate 的标准组装。"""
    if trans_fn:
        dataset = dataset.map(trans_fn)
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=True)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=False)
    return paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
