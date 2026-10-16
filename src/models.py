# -*- coding: utf-8 -*-
"""模型定义。

SemanticIndexBase        : 双塔召回底座 - CLS -> 线性降维 -> L2 归一化
SemanticIndexBatchNeg    : In-batch Negatives 对比学习训练头(仅训练时使用)
PairwiseMatching         : 交叉编码器精排 - 句对 CLS -> 线性 -> Sigmoid
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def _hidden_size(pretrained_model, default=768):
    """从预训练模型 config 中读取 hidden_size, 读取失败时退化为 768(ERNIE 系列默认)。"""
    config = getattr(pretrained_model, "config", None)
    try:
        return int(config["hidden_size"])
    except (TypeError, KeyError, ValueError, IndexError):
        return default


class SemanticIndexBase(nn.Layer):
    """双塔召回底座: 取 CLS 向量, 可选线性降维到 output_emb_size, L2 归一化。"""

    def __init__(self, pretrained_model, dropout=None, output_emb_size=None):
        super().__init__()
        self.ptm = pretrained_model
        self.output_emb_size = output_emb_size

        if output_emb_size and output_emb_size > 0:
            self.emb_reduce_linear = paddle.nn.Linear(_hidden_size(pretrained_model), output_emb_size)

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

    def get_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size and self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)

        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        """逐批产出文本向量(推理用)。"""
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)
                yield self.get_pooled_embedding(input_ids, token_type_ids=token_type_ids)

    def forward(self):
        raise NotImplementedError


class SemanticIndexBatchNeg(SemanticIndexBase):
    """In-batch Negatives 训练: 批内其他样本的文献向量作为负例。

    相似度矩阵减去对角 margin 并放大 scale 后做交叉熵, 对角线为正例位置。
    """

    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)
        self.margin = margin
        self.scale = scale

    def forward(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_position_ids=None,
        title_attention_mask=None,
    ):
        query_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)
        title_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)

        cosine_sim = paddle.matmul(query_embedding, title_embedding, transpose_y=True)

        margin_diag = paddle.full(shape=[query_embedding.shape[0]], fill_value=self.margin, dtype="float32")
        cosine_sim = (cosine_sim - paddle.diag(margin_diag)) * self.scale

        labels = paddle.arange(0, query_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])
        return F.cross_entropy(input=cosine_sim, label=labels)


class PairwiseMatching(nn.Layer):
    """交叉编码器精排: [CLS] query [SEP] 候选文献 -> Sigmoid 相关性概率。"""

    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.similarity = nn.Linear(_hidden_size(pretrained_model), 1)
        self.margin = margin

    def predict(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        sim_score = self.similarity(cls_embedding)
        return F.sigmoid(sim_score)

    def forward(
        self,
        pos_input_ids,
        neg_input_ids,
        pos_token_type_ids=None,
        neg_token_type_ids=None,
        pos_position_ids=None,
        neg_position_ids=None,
        pos_attention_mask=None,
        neg_attention_mask=None,
    ):
        _, pos_cls = self.ptm(pos_input_ids, pos_token_type_ids, pos_position_ids, pos_attention_mask)
        _, neg_cls = self.ptm(neg_input_ids, neg_token_type_ids, neg_position_ids, neg_attention_mask)

        pos_sim = F.sigmoid(self.similarity(self.dropout(pos_cls)))
        neg_sim = F.sigmoid(self.similarity(self.dropout(neg_cls)))

        labels = paddle.full(shape=[pos_cls.shape[0]], fill_value=1.0, dtype="float32")
        return F.margin_ranking_loss(pos_sim, neg_sim, labels, margin=self.margin)
