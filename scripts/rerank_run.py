# -*- coding: utf-8 -*-
"""步骤 7B/8: 用交叉编码器对某一路召回的 run 文件重排, 输出精排后的 run 文件。

对照实验用法: 每路只对 top-200 重排一次, 评估时用 evaluate.py --topk 50/100/200
截断即得三个候选数配置(N=50/100 的精排结果等价于 N=200 重排结果的前缀),
3 次重排代替 9 次, 候选集与逐配置重排完全一致。

性能选项:
  --amp           fp16 混合精度推理(T4 等 Tensor Core GPU 提速明显; 与 fp32 的
                  排序一致性需用小样本对照验证)
  --workers N     N 个预取进程并行分词并与 GPU 计算重叠(Linux fork; 0=串行,
                  Windows 本地用 0)。弱 CPU 的云 VM 建议 2

输入 run 文件格式与输出一致: 查询 \t doc_id \t 分数(输出为相关性概率)。

用法:
    python scripts/rerank_run.py --run runs/bm25_dev.run.tsv --output runs/rerank_bm25_dev.run.tsv --device gpu
    python scripts/rerank_run.py --run runs/bm25_dev.run.tsv --output out.tsv --device gpu --amp --workers 2 --batch-size 512
"""
import argparse
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import paddle
from paddlenlp.transformers import AutoModel, AutoTokenizer

from src.models import PairwiseMatching
from src.data import convert_pairwise_example
from src.utils import load_corpus


def load_run_grouped(path, topn):
    """读取 run 文件并按查询分组, 每查询按分数降序取前 topn(去重)。"""
    runs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            q, did, score = parts[0], parts[1], float(parts[2])
            if q and did:
                runs[q].append((score, did))
    out = {}
    for q, items in runs.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        seen, ranked = set(), []
        for _, did in items:
            if did in seen:
                continue
            seen.add(did)
            ranked.append(did)
            if len(ranked) >= topn:
                break
        out[q] = ranked
    return out


# ---------- 分词工作进程(Linux fork, 每进程懒加载一次 tokenizer) ----------
_W_TOKENIZER = None


def _tokenize_chunk(chunk_pairs, base_model, max_seq_len):
    """chunk_pairs: [(query, [title, title, ...])] —— 标题文本必须在主进程完成 doc_id 映射。"""
    global _W_TOKENIZER
    if _W_TOKENIZER is None:
        _W_TOKENIZER = AutoTokenizer.from_pretrained(base_model)
    trans = partial(convert_pairwise_example, tokenizer=_W_TOKENIZER,
                    max_seq_length=max_seq_len, phase="predict")
    seqs = []
    for q, titles in chunk_pairs:
        for t in titles:
            seqs.append(trans({"query": q, "title": t}))
    return seqs


def main(args):
    paddle.set_device(args.device)

    doc_ids, doc_texts = load_corpus(args.corpus)
    did2text = dict(zip(doc_ids, doc_texts))
    print("语料: {} 篇".format(len(doc_ids)))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = PairwiseMatching(pretrained)
    model.set_dict(paddle.load(args.weights))
    model.eval()
    if args.amp:
        paddle.amp.decorate(model, level="O2")
    print("rank model loaded:", args.weights, "| amp:", args.amp, "| workers:", args.workers)

    runs = load_run_grouped(args.run, args.topn)
    print("查询数:", len(runs), "| 每查询候选上限:", args.topn)

    items = list(runs.items())
    pad_id = tokenizer.pad_token_id

    def gpu_score(seqs):
        """把一个块的分词结果分批送 GPU, 返回相关性概率一维数组。seqs 每项为 (ids, types)。"""
        probs = []
        for b0 in range(0, len(seqs), args.batch_size):
            batch = seqs[b0:b0 + args.batch_size]
            L = max(len(s[0]) for s in batch)
            ids = np.full((len(batch), L), pad_id, dtype="int64")
            types = np.zeros((len(batch), L), dtype="int64")
            for i, (row_ids, row_types) in enumerate(batch):
                ids[i, :len(row_ids)] = row_ids
                types[i, :len(row_types)] = row_types
            t_ids = paddle.to_tensor(ids)
            t_types = paddle.to_tensor(types)
            with paddle.amp.auto_cast(enable=args.amp, level="O2"):
                p = model.predict(input_ids=t_ids, token_type_ids=t_types)
            probs.append(p.numpy()[:, 0])
        return np.concatenate(probs) if probs else np.zeros(0)

    chunks = [items[i:i + args.chunk_queries] for i in range(0, len(items), args.chunk_queries)]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def to_title_chunk(chunk):
        return [(q, [did2text.get(d, "") for d in dids]) for q, dids in chunk]

    n_pairs, t0 = 0, time.time()
    ex = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 0 else None
    futures = deque()

    def submit_next(idx):
        futures.append(ex.submit(_tokenize_chunk, to_title_chunk(chunks[idx]),
                                 args.base_model, args.max_seq_len))

    if ex is not None:
        for i in range(min(args.workers + 1, len(chunks))):
            submit_next(i)
        next_idx = min(args.workers + 1, len(chunks))

    with open(args.output, "w", encoding="utf-8") as f, paddle.no_grad():
        for ci, chunk in enumerate(chunks):
            if ex is not None:
                seqs = futures.popleft().result()
                if next_idx < len(chunks):
                    submit_next(next_idx)
                    next_idx += 1
            else:
                seqs = _tokenize_chunk(to_title_chunk(chunk), args.base_model, args.max_seq_len)
            probs = gpu_score(seqs)

            scores = defaultdict(list)
            pos = 0
            for q, dids in chunk:
                for d in dids:
                    scores[q].append((d, float(probs[pos])))
                    pos += 1
            for q, dids in chunk:
                ranked = sorted(scores[q], key=lambda x: -x[1])
                for d, p in ranked:
                    f.write("{}\t{}\t{:.6f}\n".format(q, d, p))
                    n_pairs += 1
            done = min((ci + 1) * args.chunk_queries, len(items))
            if done % 2000 < args.chunk_queries or ci == len(chunks) - 1:
                speed = n_pairs / max(1e-6, time.time() - t0)
                print("  {}/{} 查询, {} 对, {:.0f} 对/秒".format(done, len(items), n_pairs, speed), flush=True)

    if ex is not None:
        ex.shutdown()
    print("queries: {} | pairs: {} | 耗时 {:.0f}s".format(len(runs), n_pairs, time.time() - t0))
    print("written:", args.output)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="待重排的召回 run 文件")
    ap.add_argument("--output", required=True)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus_clean.tsv"))
    ap.add_argument("--base-model", default="ernie-3.0-medium-zh")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "rank", "rank_model_state.pdparams"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--topn", type=int, default=200, help="每查询重排的候选数上限")
    ap.add_argument("--max-seq-len", type=int, default=128, help="须与训练一致")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--chunk-queries", type=int, default=64, help="每多少个查询合并成一批处理")
    ap.add_argument("--amp", action="store_true", help="fp16 混合精度推理")
    ap.add_argument("--workers", type=int, default=0, help="分词预取进程数(0=串行)")
    main(ap.parse_args())
