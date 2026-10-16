# -*- coding: utf-8 -*-
"""hnswlib 向量索引的构建与加载。"""
import numpy as np

from paddlenlp.utils.log import logger


def build_index(dim, data_loader, model, ef_construction=100, m=100, max_elements=1_000_000, threads=16):
    """用 model 对 data_loader 中的全部文本编码, 构建 HNSW 内积索引。

    返回 (index, embeddings)。调用方负责 save_index 与 doc_ids 侧车文件的写出。
    """
    import hnswlib

    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m)
    index.set_ef(ef_construction)
    index.set_num_threads(threads)

    logger.info("start build index..........")
    all_embeddings = []
    for text_embeddings in model.get_semantic_embedding(data_loader):
        all_embeddings.append(text_embeddings.numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index.add_items(all_embeddings)
    logger.info("Total index number: {}".format(index.get_current_count()))
    return index, all_embeddings


def load_index(path, dim, ef=100):
    """加载已保存的索引; 查询前可重新设定 ef 控制召回-速度权衡。"""
    import hnswlib

    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(path)
    index.set_ef(ef)
    return index
