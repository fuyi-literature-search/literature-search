# 基于预训练大模型的中文文献检索引擎

输入一句中文查询（文献标题 / 关键词），从约 30 万篇文献中检索语义相关的文献：
**ERNIE 双塔模型 + HNSW 向量索引**先快速召回候选，再由 **ERNIE-3.0 交叉编码器**精细重排。
机器学习课程设计项目，技术路线详见开题报告（双路召回 + RRF 融合 + 统一评估已完成，结论见下）。

## 检索效果（dev 集 19,927 条查询）

| 召回路 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| ERNIE 双塔（基线复测*） | 58.26 | 69.89 | 73.88 | 77.34 | 81.57 | 63.24 | 65.79 |
| BM25（jieba，k1=1.5，b=0.75） | 71.17 | 83.07 | 86.02 | 88.48 | 90.79 | 76.28 | 78.65 |
| RRF 融合（c=60，depth=200） | 64.83 | 79.21 | 84.41 | 88.65 | 92.24 | 71.11 | 74.30 |
| **RRF 融合（c=5，调参后）** | 66.43 | — | **87.58** | — | **92.22** | 73.84 | 77.19 |

> *基线复测：原索引未随权重分发，本表 ERNIE 行为本机重建索引后的结果，与原报告基线
> （58.19/69.73/73.68/77.07/81.07，MRR 63.14，nDCG 65.67）偏差：R@1~@10 与 MRR/nDCG ≤0.2 点，
> R@20 +0.27、R@50 +0.50，验证重建有效。
> BM25 行 29 个纯英文/罕见词查询（语料中无该词项）无召回结果已排除，n=19,898。
> 早期版本因语料重复文献重复计数，Recall@5~@50 虚高约 4 个点，以上为修正后数值。

RRF 常数 c 敏感性（Recall@10 / Recall@50）：c=1: 87.42/92.22；c=5: **87.58**/92.22；c=10: 87.49/92.24；
c=20: 86.71/**92.29**；c=60: 84.41/92.24；c=120: 83.32/91.64；c=300: 82.81/91.32。
小 c 减弱排名稀释、浅层反超 BM25；曲线在 c∈[1,10] 平坦，取 c=5（注意: dev 兼作调参集，为已知局限）。

两路互补性（Recall@200，`scripts/union_coverage.py`）：BM25 93.61 / ERNIE 87.00 / 并集 95.54 /
RRF(c=60) 94.80 —— 互补上限 +1.93，RRF 兑现 +1.19，截断损失 0.75。

五路召回对照（`tfidf_runner.py` / `bm25_prf.py` / `build_index.py --weights ""` 零训练臂，
`summarize_recall.py` 汇总；完整表含配对 bootstrap 置信区间，见 `runs/recall_comparison.md`）：

| 召回路 | Hit@10 | MRR@10 | nDCG@10 | R@200 |
|---|---|---|---|---|
| TF-IDF（与 BM25 同分词） | 82.68 | 71.18 | 73.95 | 92.97 |
| **BM25** | **86.02** | **76.28** | **78.65** | **93.61** |
| BM25+PRF（反馈10篇/扩展10词） | 79.94 | 48.19 | 55.85 | 92.08 |
| ERNIE 零训练底座 | 0.00 | 0.00 | 0.00 | 0.10 |
| ERNIE 微调双塔（即上表基线复测） | 73.89 | 63.25 | 65.80 | 86.92 |

三条结论（均经配对 bootstrap 检验显著）：① BM25 的词频饱和与文档长度归一化打分相对朴素
TF-IDF 全面占优（MRR@10 +5.11）；② 本场景查询即文献标题、正例为精确匹配，朴素 PRF 的
扩展词反而把精确匹配挤出前排（MRR@10 −28.09），查询扩展在此类语料上不适用；③ 预训练
底座不微调直接编码，检索能力接近随机（R@200 与 29.2 万篇随机取 200 篇的命中期望同量级），
对比学习微调是检索能力的唯一来源——这是本项目训练环节价值的直接量化。

## 检索效率（`scripts/benchmark_latency.py`，Intel 12 代 + RTX 3060 Laptop）

| 阶段 | 构建/加载 | 延迟 P50 / P95 | 吞吐 | 峰值内存 |
|---|---|---|---|---|
| BM25（单线程 CPU） | 索引构建 69.7s | 7.7ms / 42.3ms | 60.8 QPS | 337MB |
| ERNIE 召回（GPU 单查询端到端） | 模型+索引加载 | 14.4ms / 21.4ms | 57.4 QPS | 1.6GB |
| 精排（GPU，batch 64，seq 128） | 模型加载 | 3.86ms/对 | 259 对/秒 | 1.0GB |

HNSW ef 实验（`scripts/benchmark_ef.py`，500 查询，以 ef=1000 为近似精确基准）：
ef=100 时 top-200 重合率 98.45%（0.08ms/查询），ef=400 时 99.6% —— 近似误差可忽略。

## 精排对照（最终排序，5,000 查询子样本，seed=42）

召回 top-200 经交叉编码器重排后的最终指标（`scripts/rerank_run.py` + `evaluate.py --topk`；
top-10 类指标与 N 截断无关，故各行同值；完整 N=50/100/200 对照见 `runs/rerank_sub5k_*.metrics.json`）：

| 召回路 + 精排 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| ERNIE 单路 | 64.91 | 77.42 | 81.08 | 70.31 | 72.91 |
| BM25 单路 | 65.33 | 79.12 | 83.76 | **71.34** | **74.33** |
| RRF(c=5) 融合 | 64.31 | 78.54 | 83.52 | 70.51 | 73.64 |

> 注：精排对照仅覆盖 c=5（调参后正式配置）；c=60 精排路未做（候选池 `runs/sub5k_rrf.run.tsv` 已备）。

**核心结论（开题研究问题的答案）**：融合的价值止步于召回层。召回层 RRF(c=5) 显著优于
BM25（R@10 87.58 vs 86.02，R@50 +1.4）；但经同一交叉编码器精排后，BM25 单路在 MRR@10
（+0.83）与 nDCG@10（+0.70）上显著反超融合路，R@10 差异不显著（-0.24，95%CI 跨零）。
归因：融合候选池 Recall@200 更高（+1.2）但引入了向量路的语义近义干扰项，交叉编码器难以
完全排除，top 名次被稀释；BM25 池词面紧凑、干扰项更易被打分排除。与开题"若融合未优于
最强单路，检查召回覆盖、融合截断和精排"的预判一致。另注：精排对 ERNIE 路提升明显
（R@10 73.9→81.1），对 BM25 路反而略降（86.0→83.8，词面精确命中被语义重排挤出 top-10）。

## 案例分析（按术语缩写 / 型号差异 / 近义表达，开题五.3）

基于 dev 全量 run 的两路互补查询集（BM25@10 未命中而 ERNIE@10 命中 vs 反向）：

- **BM25 失手、ERNIE 救回（664 条）**：89% 为纯中文查询，典型形态是**近义改写与语序重组**——
  如查询"中国经济增长和失业问题的实证研究"对应文献"……经济增长……失业……实证研究"（词面不重合）、
  "0~3 岁大运动"（口语化短表达）、"2.25Cr1Mo0.25V 晶界"（钢材牌号连写导致分词碎化）；
  另 11% 含英文/拼音（如 `Apexification`），其中 29 条语料无该词项，BM25 零结果，属词法检索盲区。
- **ERNIE 失手、BM25 救回（3,057 条）**：多为**布尔式检索串与型号/缩写精确匹配**——
  如 `((白血病AND(乳酸脱氢酶AORLDHA)))`、`(HIFORKDM6BORCaveolin)肾康丸`、`(S)-1,2-丙二醇合成`，
  词面精确命中是唯一有效信号，向量语义反而引入干扰。
- 结论与开题"三、已有方法的局限"的预判一致：词法与语义两路失败的查询形态互补，
  这是融合在召回层获得 +1.9 点互补收益的来源；但干扰项同样随语义路进入精排候选池，
  构成最终排序层融合被反超的归因（见"核心结论"）。

## 局限与说明

1. **权重划分**：召回/精排权重沿用原始划分训练（同学提供）；查询分组划分（`build_split.py`）
   已生成去泄漏训练集（剔除召回 1.0%、精排 7.9% 与 dev/test 近重复样本），正式重训留作后续。
2. **精排对照规模**：为控制推理成本，最终排序对照采用 dev 中四路均有结果的 19,898 条查询里
   按 seed=42 抽取的 5,000 条子样本（配对比较，95% CI 约 ±1.3 点）；召回层指标均为全量 19,927 条。
3. **test 集不可用**：895/998 正例不在语料（开题已知问题），全部实验统一用 dev。
4. **调参集**：RRF 常数 c 在 dev 上调得（dev 兼作评估集），敏感性曲线在 c∈[1,10] 平坦，结论稳健。
5. **未标注 ≠ 不相关**：抽样核验（`runs/sample_unlabeled_review.tsv`，30 查询 × Top-5 = 150 条）
   中约 13.3%（20/150）的"未标注正例"结果经人工判断实际相关——真实 Recall 存在系统性低估，
   各路 Recall 均受此影响。

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
│   ├── build_split.py       2B. 查询分组划分(MinHash+LSH 近重复聚类, 剔除与 dev/test 同组的训练查询)
│   ├── train_recall.py      3. 微调 ERNIE-1.0 双塔召回模型(In-batch Negatives)
│   ├── build_index.py       4. 语料全量编码 + 构建 HNSW 索引
│   ├── run_recall.py        5. 批量召回, 输出统一 run 文件(查询\tdoc_id\t分数)
│   ├── bm25_runner.py       5B. BM25 关键词召回(批量出 run 文件; 核心在 src/bm25.py)
│   ├── tfidf_runner.py      5C. TF-IDF 词法对照基线(同分词规则, 批量出 run 文件)
│   ├── bm25_prf.py          5D. BM25+伪相关反馈(PRF)对照路(两轮检索+查询扩展)
│   ├── summarize_recall.py  5E. 五路召回对照实验汇总(生成 runs/recall_comparison.md)
│   ├── train_rank.py        6. 训练交叉编码器精排模型(pairwise margin ranking loss)
│   ├── predict_rank.py      7. 精排模型对测试集打分
│   ├── rerank_run.py        7B. 对召回 run 文件重排(每路重排 top-200 一次, 配合 evaluate --topk 得 N=50/100/200)
│   ├── benchmark_latency.py 9. 各阶段效率测试(延迟 P50/P95/P99、吞吐、内存与构建耗时)
│   └── search.py            8. 端到端交互式搜索演示(召回+精排)
├── src/                     共享库(所有脚本复用, 无重复实现)
│   ├── models.py            SemanticIndexBase / SemanticIndexBatchNeg / PairwiseMatching
│   ├── data.py              数据读取、文本转换、批处理整理(collate)
│   ├── ann.py               hnswlib 索引构建与加载
│   ├── bm25.py              BM25 倒排索引与打分(bm25_runner 与 webapp 共用)
│   └── utils.py             文本规范化、doc_id、语料加载
├── eval/                    统一评估工具
│   ├── metrics.py           Recall@N / Hit@K / MRR@K / nDCG@K + 配对 bootstrap 置信区间
│   ├── evaluate.py          评估入口(--baseline 可对比两组实验)
│   └── rrf.py               RRF 多路排名融合(供混合召回实验)
├── datasets/
│   ├── recall/              corpus.csv(30万语料) + train/dev/test 配对
│   └── rank/                train/dev/test 三元组与句对
├── webapp/
│   └── search.py            单文件网页搜索服务(内嵌页面; bm25/ernie/rerank 三引擎)
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

# 2) 构建向量索引(已有 index/ 可跳过; 不传 --weights 时自动用默认权重路径)
python scripts/build_index.py --device gpu

# 3) 批量召回 -> run 文件
python scripts/run_recall.py --topk 200            # 先加 --limit 200 快速验证

# 4) 评估
python eval/evaluate.py --qrels data/qrels_dev.tsv --run runs/ernie_dev.run.tsv --recall-ks 1 5 10 20 50

# 5) 交互式搜索体验
python scripts/search.py --query "热处理对尼龙6结晶行为的影响"
```

## 网页版搜索演示（单文件 `webapp/search.py`）

```bash
python webapp/search.py                    # BM25 引擎(纯 CPU), 自动打开浏览器
python webapp/search.py --engine rerank    # BM25 召回 + ERNIE-3.0 精排(需 paddle)
python webapp/search.py --engine ernie     # 向量召回(需 paddle 与 index/)
```

浏览器访问 http://127.0.0.1:8000 ，支持关键词高亮、相关度条、返回条数/引擎切换与
样例查询。页面由脚本内嵌，无需任何前端构建；除 bm25 引擎外需 paddle 环境。

精排部分：

```bash
python scripts/train_rank.py --device gpu          # 训练(已有 weights/rank 可跳过)
python scripts/predict_rank.py                     # 测试集打分
```

混合召回实验（BM25 + ERNIE + RRF）：

```bash
# BM25 路: 输出 runs/bm25_dev.run.tsv, 格式与 ERNIE 路一致(纯 CPU, 无需权重)
python scripts/bm25_runner.py --topk 200
# RRF 融合 + 与 ERNIE 单路对照(配对 bootstrap 显著性)
python eval/rrf.py --runs runs/bm25_dev.run.tsv runs/ernie_dev.run.tsv --depth 200 --out runs/rrf_dev.run.tsv
python eval/evaluate.py --qrels data/qrels_dev.tsv --run runs/rrf_dev.run.tsv --baseline runs/ernie_dev.run.tsv
```

## 数据与权重（超过 GitHub 100MB 限制，不入 git 仓库）

以下文件不进 git，由组内 QQ / 网盘直接分发；拿到文件后按下表位置放置即可
（文件名与脚本默认路径一致，无需改名。若以后想集中管理，也可用 GitHub
Releases 页面上传，单文件上限 2GB）。

| 文件 | 放置位置 | 大小 | 也可通过 |
|---|---|---|---|
| train_pairwise.csv | datasets/rank/ | 456MB | 课程数据包 literature_search_data |
| recall_model_state.pdparams（召回权重） | weights/recall/ | 382MB | `scripts/train_recall.py` 重新训练 |
| rank_model_state.pdparams（精排权重） | weights/rank/ | 288MB | `scripts/train_rank.py` 重新训练 |
| final_index.bin | index/ | 514MiB* | `scripts/build_index.py` 重新生成（GPU 约 15 分钟） |

> *原索引 504MiB 未随仓库分发；本表为本机用召回权重重建的索引（m=100, ef_construction=100），
> 重建后 ERNIE 基线复测与原报告偏差 ≤0.2 点（见「检索效果」表注）。

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

- [x] BM25 召回路（jieba 分词，保留术语/缩写/数字），输出统一 run 文件（`scripts/bm25_runner.py`）
- [x] RRF 参数 c 在调参集上敏感性分析（c∈{1,5,10,20,60,120,300}；取 c=5，浅层反超 BM25）
- [x] 端到端 P50/P95 延迟与各阶段耗时统计（`scripts/benchmark_latency.py`，三阶段数据见「检索效率」表）
- [x] 查询级分组划分（按查询及近重复查询分组，避免泄漏）（`scripts/build_split.py`；42.3 万唯一查询分组后剔除：召回训练 1.0%、精排训练 7.9% 与 dev/test 近重复）
- [ ] 困难负例挖掘训练精排模型（可选扩展）
