# -*- coding: utf-8 -*-
"""文献语义检索引擎 - 共享库。

模块:
    models : 双塔召回模型(ERNIE)与交叉编码器精排模型(PairwiseMatching)
    data   : 数据读取 / 文本转换 / 批处理整理
    ann    : hnswlib 向量索引构建与加载
    utils  : 文本规范化与稳定 doc_id
"""
