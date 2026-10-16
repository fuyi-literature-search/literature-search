# 基于预训练大模型的中文文献检索引擎

输入一句中文查询（文献标题 / 关键词），从约 30 万篇文献中检索语义相关的文献：
**ERNIE 双塔模型 + HNSW 向量索引**先快速召回候选，再由 **ERNIE-3.0 交叉编码器**精细重排。
机器学习课程设计项目，技术路线详见开题报告（双路召回 + RRF 融合为进行中工作）。

## 检索效果（dev 集 19,927 条查询）

| Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| 58.19 | 69.73 | 73.68 | 77.07 | 81.07 | 63.14 | 65.67 |

> 口径说明：run 内按 doc_id 去重后统计；dev 正例 100% 命中语料，全部查询可评价。
> 注意：早期版本因语料重复文献重复计数，Recall@5~@50 虚高约 4 个点，以上为修正后数值。

## 系统架构

```
                 ┌─────────────── 离线 ───────────────┐
 corpus.csv ──> build_corpus ──> corpus_clean.tsv ──> build_index ──> HNSW索引+doc_ids
                                        │                (weights/recall)
                                        └──> build_qrels ──> qrels_dev.tsv
                 ┌─────────────── 在线 ────────────────┐
 查询 ──> run_recall(ERNIE编码+HNSW检索) ──> runs/*.run.tsv ──> eval/evaluate
              │                                                    │
              └──> search.py(召回Top50 → 精排Top10 交互演示) <── weights/rank
```

## 目录结构

```
├── scripts/                 流水线脚本(按步骤编号, 依次执行即可复现)
│   ├── build_corpus.py      1. 语料去重 + 稳定 doc_id(sha1 前 16 位)
│   ├── build_qrels.py       2. 查询-正例映射 qrels(支持多正例, 报告可评价查询数)
│   ├── train_recall.py      3. 微调 ERNIE-1.0 双塔召回模型(In-batch Negatives)
│   ├── build_index.py       4. 语料全量编码 + 构建 HNSW 索引
│   ├── run_recall.py        5. 批量召回, 输出统一 run 文件(查询\tdoc_id\t分数)
│   ├── train_rank.py        6. 训练交叉编码器精排模型(pairwise margin ranking loss)
│   ├── predict_rank.py      7. 精排模型对测试集打分
│   └── search.py            8. 端到端交互式搜索演示(召回+精排)
├── src/                     共享库(所有脚本复用, 无重复实现)
│   ├── models.py            SemanticIndexBase / SemanticIndexBatchNeg / PairwiseMatching
│   ├── data.py              数据读取、文本转换、批处理整理(collate)
│   ├── ann.py               hnswlib 索引构建与加载
│   └── utils.py             文本规范化、doc_id、语料加载
├── eval/                    统一评估工具
│   ├── metrics.py           Recall@N / Hit@K / MRR@K / nDCG@K + 配对 bootstrap 置信区间
│   ├── evaluate.py          评估入口(--baseline 可对比两组实验)
│   └── rrf.py               RRF 多路排名融合(供混合召回实验)
├── datasets/
│   ├── recall/              corpus.csv(30万语料) + train/dev/test 配对
│   └── rank/                train/dev/test 三元组与句对
├── weights/                 微调权重(gitignore, 见「数据与权重」)
├── index/                   HNSW 索引(gitignore)
├── data/                    派生数据(可由脚本重新生成, gitignore)
└── runs/                    召回 run 文件与指标存档
```

## 环境要求

- Python 3.8 – 3.10
- 依赖见 [requirements.txt](requirements.txt)（PaddlePaddle 2.6 + PaddleNLP 2.6 + hnswlib）
- 训练建议 GPU（CPU 也可运行，30 万语料建索引约需数小时）；评估与数据脚本纯 CPU 秒级

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 0) 数据准备(已完成可跳过): 语料去重 + 多正例 qrels
python scripts/build_corpus.py
python scripts/build_qrels.py

# 1) 训练召回模型(已有 weights/recall 可跳过)
python scripts/train_recall.py --device gpu --save-steps 100

# 2) 构建向量索引(已有 index/ 可跳过)
python scripts/build_index.py --weights weights/recall/model_state.pdparams --device gpu

# 3) 批量召回 -> run 文件
python scripts/run_recall.py --topk 200            # 先加 --limit 200 快速验证

# 4) 评估
python eval/evaluate.py --qrels data/qrels_dev.tsv --run runs/ernie_dev.run.tsv --recall-ks 1 5 10 20 50

# 5) 交互式搜索体验
python scripts/search.py --query "热处理对尼龙6结晶行为的影响"
```

精排部分：

```bash
python scripts/train_rank.py --device gpu          # 训练(已有 weights/rank 可跳过)
python scripts/predict_rank.py                     # 测试集打分
```

混合召回实验（BM25 + ERNIE + RRF，进行中）：

```bash
# BM25 路(bm25_runner.py 待接入): 输出 runs/bm25_dev.run.tsv, 格式与 ERNIE 路一致
python eval/rrf.py --runs runs/bm25_dev.run.tsv runs/ernie_dev.run.tsv --depth 200 --out runs/rrf_dev.run.tsv
python eval/evaluate.py --qrels data/qrels_dev.tsv --run runs/rrf_dev.run.tsv --baseline runs/ernie_dev.run.tsv
```

## 数据与权重（超过 GitHub 100MB 限制，通过 Releases 分发）

以下文件不入 git 仓库，统一放在本仓库的 **Releases** 页面（组员下载后放到对应位置）：
打开 https://github.com/&lt;你的GitHub用户名&gt;/literature-search/releases 选择最新 Release 下载。

| 文件 | 放置位置 | 大小 | 也可通过 |
|---|---|---|---|
| train_pairwise.csv | datasets/rank/ | 456MB | 课程数据包 literature_search_data |
| recall_model_state.pdparams（召回权重） | weights/recall/ | 382MB | `scripts/train_recall.py` 重新训练 |
| rank_model_state.pdparams（精排权重） | weights/rank/ | 288MB | `scripts/train_rank.py` 重新训练 |
| final_index.bin | index/ | 528MB | `scripts/build_index.py` 重新生成（约数小时，需 GPU） |

> 未下载权重/索引时，可先跑纯数据链路：`build_corpus.py` → `build_qrels.py`；排序训练集
> 缺失时 `train_rank.py` 需先补数据。模型底座（ernie-1.0、ernie-3.0-medium-zh）由
> PaddleNLP 自动下载；网络受限时设置 `export HF_ENDPOINT=https://hf-mirror.com`。

## run 文件格式约定

所有检索路（ERNIE / BM25 / RRF）统一输出 TSV：`查询 \t doc_id \t 分数`。
评估器按分数重排、按 doc_id 去重（保留最优名次）；`--baseline` 输出查询级配对差值与
95% bootstrap 置信区间（seed=42, 1000 次重采样），直接用于报告显著性分析。

## 已知问题与实验规范

1. **test.csv 与语料不匹配**：998 条测试配对中 895 条正例不在语料内，不能用于召回评估；
   召回实验统一用 dev 集（19,927 条全部可评价），`build_qrels.py` 会自动报告排除情况。
2. **多正例**：dev 中 11 个查询有多个正例，qrels 已按集合处理；Recall@N 按比例计。
3. **对照实验公平性**：三组配置共用同一语料、同一测试查询、同一精排模型与输入长度(128)；
   变量只有召回策略与候选数 N(50/100/200)，`run_recall.py --topk` 与 `evaluate.py --recall-ks` 控制。
4. **训练/推理一致性**：精排 tokenizer、输入长度(128)训练与推理必须一致（曾因不一致引入 bug）。

## 组员分工

| 成员 | 分工 |
|---|---|
| 刘铭 | 数据整理、查询划分与实验基准 |
| 范航 | ERNIE 双塔召回与 HNSW 索引 |
| 倪阳 | BM25 索引、关键词召回与 RRF 融合 |
| 黄亦诚 | 交叉编码器精排与效率测试 |
| 焦恒敏 | 实验评价、系统集成与报告统稿 |

## TODO

- [ ] BM25 召回路（jieba 分词，保留术语/缩写/数字），输出统一 run 文件
- [ ] RRF 参数 c 在调参集上敏感性分析
- [ ] 端到端 P50/P95 延迟与各阶段耗时统计
- [ ] 查询级分组划分（按查询及近重复查询分组，避免泄漏）
- [ ] 困难负例挖掘训练精排模型（可选扩展）
