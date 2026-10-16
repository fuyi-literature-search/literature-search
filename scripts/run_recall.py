# -*- coding: utf-8 -*-
"""步骤 5/8: 批量召回 - 对查询文件逐条检索 Top-K, 输出统一 run 文件。

run 文件格式(TSV): 查询 \t doc_id \t 分数, 供 eval/evaluate.py 与 eval/rrf.py 使用。
查询文件每行第一列为查询文本(自动去重); 文档分数 = 1 - 内积距离(即余弦相似度)。

用法:
    python scripts/run_recall.py --limit 200          # 先跑 200 条快速验证
    python scripts/run_recall.py                      # dev 全量 19,927 条
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
from src.data import convert_bi_example, make_batchify, create_dataloader, read_queries
from src.ann import load_index


def main(args):
    paddle.set_device(args.device)

    doc_ids_path = os.path.join(args.index_dir, "doc_ids.json")
    if os.path.isfile(doc_ids_path):
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
    else:
        doc_ids = None
        print("警告: 未找到 doc_ids.json, run 文件将使用行号作为 doc_id")

    index = load_index(os.path.join(args.index_dir, "final_index.bin"), dim=args.output_emb_size, ef=args.ef)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained = AutoModel.from_pretrained(args.base_model)
    model = SemanticIndexBase(pretrained, output_emb_size=args.output_emb_size)
    if args.weights:
        model.set_dict(paddle.load(args.weights))
        print("loaded weights:", args.weights)

    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print("查询数:", len(queries))

    trans_fn = partial(convert_bi_example, tokenizer=tokenizer, max_seq_length=args.max_seq_len)
    query_ds = MapDataset(queries)
    loader = create_dataloader(
        query_ds, mode="infer", batch_size=args.batch_size,
        batchify_fn=make_batchify("bi_infer", tokenizer), trans_fn=trans_fn)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        cursor = 0
        for batch_embedding in model.get_semantic_embedding(loader):
            recalled_idx, distances = index.knn_query(batch_embedding.numpy(), args.topk)
            for row in range(len(distances)):
                query_text = queries[cursor]["query"]
                cursor += 1
                for idx, doc_row in enumerate(recalled_idx[row]):
                    score = 1.0 - float(distances[row][idx])
                    did = doc_ids[doc_row] if doc_ids else "line{}".format(doc_row)
                    f.write("{}\t{}\t{}\n".format(query_text, did, score))
                    n_written += 1

    print("queries: {} | lines: {}".format(cursor, n_written))
    print("written:", args.output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "index"))
    ap.add_argument("--queries", default=os.path.join(ROOT, "datasets", "recall", "dev.csv"))
    ap.add_argument("--base-model", default="ernie-1.0")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "recall", "recall_model_state.pdparams"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--output-emb-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=200,
                    help="每查询召回条数; 对照实验需要 N=50/100/200, 故默认 200")
    ap.add_argument("--ef", type=int, default=100, help="查询时的 ef 参数(召回-速度权衡)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条查询, 0 表示全部")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "ernie_dev.run.tsv"))
    main(ap.parse_args())
