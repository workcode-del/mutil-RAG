# 统一评测

公开 benchmark 与自定义证据图共用同一套检索系统、指标和报告格式。部署与建库见 [DEPLOYMENT.md](DEPLOYMENT.md)，算法定义见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 公开数据集

一条命令完成数据准备、Dense 索引、系统矩阵和汇总：

```bash
paper-rag benchmark all \
  --datasets peerqa mmdocrag \
  --root data/benchmarks \
  --config configs/default.yaml \
  --setting 20
```

只有包含真实结构边的数据集才应使用 `--train-graph-index` 做 HGT/R-GCN
对照。例如 PeerQA 可直接按隔离划分训练：

```bash
paper-rag benchmark all --datasets peerqa --root data/benchmarks \
  --config configs/default.yaml --train-graph-index --graph-model hgt
```

也可分阶段运行：

```bash
paper-rag benchmark prepare --datasets peerqa mmdocrag --root data/benchmarks
paper-rag benchmark train --datasets peerqa mmdocrag --root data/benchmarks \
  --config configs/default.yaml
paper-rag benchmark run --datasets peerqa mmdocrag --root data/benchmarks \
  --config configs/default.yaml --split test --systems dense full \
  --hgt-artifacts outputs/benchmark_graph
```

R-GCN 使用与 HGT 相同的基础向量、query/gold 对、同类型 hard negative、关系 InfoNCE、训练轮数和检索后端。建议独立训练两个产物后在同一测试划分比较：

```bash
paper-rag benchmark train --datasets peerqa --root data/benchmarks \
  --config configs/default.yaml --graph-model rgcn \
  --graph-output-root outputs/benchmark_rgcn
paper-rag benchmark run --datasets peerqa --root data/benchmarks \
  --config configs/default.yaml --split test --systems rgcn \
  --rgcn-artifacts outputs/benchmark_rgcn
```

已有下载和解析结果会复用；`--force` 重做准备，`--reindex` 重建 embedding 缓存。缓存 sidecar 校验图哈希和 embedding 配置哈希，模型或 query instruction 改变时自动失效。benchmark 在 NPZ 上做精确 cosine 检索，不依赖 Qdrant；在线服务仍使用配置的向量库。

### 数据集与口径

| 数据集 | 检索单位与范围 | Recall@K | 官方读取文件 | 当前状态 |
|---|---|---|---|---|
| PeerQA | 官方句子；按论文限制候选 | 1/3/5/10 | `qa.jsonl`、`papers.jsonl` | 可 prepare/train/test；固定 HF revision |
| MMDocRAG | 每题官方 text/image quotes | 1/3/5/10/15/20 | `dev_{15,20}.jsonl`、`evaluation_{15,20}.jsonl`、`images.zip` | 可 prepare/test；quote 图无结构边，不宜作为 HGT/R-GCN 图对照 |
| M3DocVQA | 官方为 open-domain PDF 文档/页面 | 1/3/5/10 | `multimodalqa/MMQA_dev.jsonl`、`dev_doc_ids.json`、`splits/pdfs_dev` | 官方数据没有 gold 页码；当前仅兼容派生页标注，不能报官方结果 |
| MMLongBench-Doc | 单文档页面，支持多页 gold | 1/3/5/10 | `data/samples.json`、`data/documents/*.pdf` | 可 prepare/test；内部文档隔离划分可训练投影，但当前页图无结构边 |
| MultimodalQA | LILaC open-domain text/table/image 组件 | 1/3/5/10 | 官方原始文件为 `MultiModalQA_dev.jsonl.gz`、`texts.jsonl.gz`、`tables.jsonl.gz`、`images.jsonl.gz`、`images.zip` | 当前读取 LILaC 派生 Parquet，不等同官方原始协议；仅作派生实验 |
| SPIQA | 论文内 figure/table 图像；test-A 为官方测试 | 1/3/5/10 | `SPIQA_train.json`、`SPIQA_val.json`、`SPIQA_testA.json`、对应图片 ZIP | 可 prepare/train/test，含真实 `caption_of` 边 |

显式 `--ranking-k` 覆盖默认值。`comparison.csv` 导出整体与 Sentence/Figure/Table 分模态 Recall、Evidence F1、结构、预算和延迟指标。

#### PeerQA 与 MMDocRAG

- PeerQA 直接使用官方 `papers.jsonl` 的句子和 `idx` 构图，按 `paper_id` 稳定划分；`--peerqa-download-pdfs` 仅用于 MinerU 全文扩展实验。
- MMDocRAG 默认 `setting=20`，把 quote 转成 Sentence/Figure，并用 `candidate_node_ids` 保证所有方法使用相同候选；dev 用于内部 train/dev，evaluation 为 test。
- MMDocRAG quote 图不虚构全文关系，因此 `relation_triples` 可能为 0。下载 PDF 的 `--mmdocrag-download-pdfs` 目前不负责把全文节点与 quote gold 对齐。
- 自动下载均固定官方 revision，并把 `source_revision` 写入 `prepare_report.json`；传入本地 `--dataset-source` 时该值为空，由使用者记录本地快照版本。

#### 新增的跨论文多模态数据

MMLongBench-Doc 自动下载 `samples.json` 和实际使用的 PDF，只保留带 `evidence_pages` 的问题。官方 gold 页号从 1 开始，节点沿用相同页码：

```bash
paper-rag benchmark prepare --datasets mmlongbench_doc --root data/benchmarks
```

`--max-documents 5` 只用于冒烟测试，报告会标记为 `partial_documents`。已有官方快照可通过 `--dataset-source "mmlongbench_doc=/data/MMLongBench-Doc"` 使用。

MultimodalQA 默认下载 LILaC 发布的 `JoohyungYun/multimodalqa_doc`，不是 AllenAI 原始发布包。当前 Hugging Face 快照中的 `dev.parquet`、`text.parquet`、`table.parquet`、`image.parquet` 和 `image_dump.parquet` 可直接读取，无需手工运行快照附带的 `load.py`。Parquet 还原严格沿用该脚本的字段约定：三类组件读取 `doc_title`、`component_id` 和 `component`，图片字节读取 `image_name` 与 `byte_data`；`heading_path`、`hyperlinks` 和 `label_id` 会保留到图节点属性。约 5 GB 的图片 Parquet 按小批流式恢复，不会像其 pandas 脚本一样整表载入内存。空图、零可用样本或任一模态完全丢失都会直接报错，不再写出看似成功的空数据。适配器也继续兼容 LILaC 的 `QAs_dev_labeled.json`、`parsed_documents`、`image_components` 目录或 ZIP。text、table、image 分别建成 Sentence、Table、Figure：

```bash
paper-rag benchmark prepare --datasets multimodalqa --root data/benchmarks
```

报告会写入 `evaluation_scope=lilac_component_dev_snapshot` 和
`official_benchmark=false`。即便缺图和缺 gold 已清零，运行或训练仍需显式
`--allow-partial`，论文中应称为“LILaC 组件级 MultimodalQA 派生设置”。

SPIQA 使用官方 train、val、test-A 划分；`--split official` 固定映射到 test-A，test-A 不参与 HGT 训练。每篇论文的 figure/table 是公平候选集，caption 单独建点，并且只添加数据明确给出的 `caption_of`：

```bash
paper-rag benchmark prepare --datasets spiqa --root data/benchmarks
paper-rag benchmark train --datasets spiqa --root data/benchmarks \
  --config configs/default.yaml
paper-rag benchmark run --datasets spiqa --root data/benchmarks \
  --config configs/default.yaml --split official --systems dense full \
  --hgt-artifacts outputs/benchmark_hgt
```

官方 train/val 图片压缩包约 32 GB，诊断时可先下载并解压所需子集，再通过 `--dataset-source "spiqa=/data/SPIQA"` 指向本地快照。SPIQA 能直接验证 QA→Figure/Table 检索监督和 `caption_of` 关系监督；它没有正文句子到图表的人工 `refers_to` 标注，因此本评测不会构造该关系，也不能单独证明正文引用关系学习有效。

M3DocVQA 官方代码从 MultiModalQA 的 `MMQA_dev.jsonl` 读取 `supporting_context[].doc_id`，并从 `splits/pdfs_dev` 动态渲染页面；官方标注没有 gold 页码。项目不把“支持文档的所有页面”伪造为 gold，也不静默使用第三方镜像。若另有可追溯的派生页标注，兼容输入需包含 `M3DocVQA_dev_labeled.json` 与 `pdf_pages/dev`：

```bash
paper-rag benchmark prepare --datasets m3docvqa \
  --dataset-source "m3docvqa=/data/M3DocVQA"
```

该适配器把报告标为 `derived_page_labeled_snapshot` 和
`official_benchmark=false`；运行或训练时必须显式加 `--allow-partial`，结果只能作为派生诊断实验。

M3DocVQA 与 MMLongBench-Doc 的官方检索单位是整页，所以页面建为 Figure；`required_modalities` 只记录证据来源，不能据此把整页虚构成 Table。Table 的直接分模态评测来自 MultimodalQA 组件或 MinerU 全文图。

新增集合目前使用 dev/all 标注；无训练方法的 `--split official` 对应 `all`。训练 HGT 后必须报告内部 held-out test，并通过训练/评测 query 重叠检查。

## 2. 自定义数据

每行一个 JSON 样本：

```json
{
  "query_id":"q1",
  "paper_id":"paper1",
  "query":"What is the main contribution?",
  "answer":"reference answer",
  "relevant_node_ids":["paper1:sentence:8:1"],
  "candidate_node_ids":["paper1:sentence:8:1","paper1:sentence:9:0"],
  "required_modalities":["text"]
}
```

`query` 和非空 `relevant_node_ids` 必需。`paper_id/paper_ids` 限制论文范围，`candidate_node_ids` 进一步限制候选且必须包含全部 gold；`answer` 只在启用生成时使用。`required_modalities` 是用于分层报告的 gold 元数据，不参与检索选择或 EC-BFR 槽位评分，避免把答案模态泄漏给某一种方法。

建立索引并运行单次实验：

```bash
paper-rag index data/parsed/evidence_graph.json \
  --embedding-cache data/cache/base_embeddings.npz \
  --config configs/default.yaml

python scripts/evaluate_retrieval.py data/eval/test.jsonl \
  --graph data/parsed/evidence_graph.json \
  --candidate-backend embedding \
  --retrieval-method ec_bfr \
  --graph-artifacts outputs/hgt \
  --ranking-k 1 3 5 10 \
  --output outputs/eval/full.json
```

默认 `--scope sample` 使用样本论文范围；`--scope corpus` 才是跨论文全库检索。BM25 使用 `--candidate-backend bm25 --retrieval-method top_k --disable-reranker`。多个报告可用 `scripts/compare_evaluations.py` 汇总。

## 3. 对比系统

| 方法 | 候选 | 结构处理 | 其他 |
|---|---|---|---|
| `bm25` | BM25 | top-k | 无 Reranker |
| `dense` | Dense | top-k | 无 Reranker |
| `dense_reranker` | Dense | top-k | 多模态 Reranker |
| `one_hop` | Dense | 一跳扩展 | — |
| `ppr` | Dense | PPR | — |
| `pcst` | Dense | PCST | — |
| `pcst_closure` | Dense | PCST + 证据闭包 | — |
| `ec_bfr` | Dense | 闭包 + 硬预算森林 | 无 HGT |
| `ec_bfr_reranker` | Dense | EC-BFR | Reranker |
| `rgcn` | Dense + R-GCN | EC-BFR | Reranker，需 R-GCN 训练产物 |
| `full` | Dense + HGT | EC-BFR | Reranker，需训练产物 |

比较时除目标消融项外，应保持图、候选范围、模型、预算、top-k 和评测 split 相同。

## 4. 指标与输出

- 排序：MRR、MRR@10、Recall@K、nDCG@K、Joint Recall@K；
- 证据集：macro/micro Evidence Precision、Recall、F1；
- 分模态：Sentence、Figure、Table、Caption、ChartData 的 Recall@K 与 Evidence F1；
- 结构：Closure Validity、Dependency Completeness；
- 效率：Budget Violation、Evidence Cost、Selected Nodes、检索与 query embedding 延迟；
- 可选生成：Exact Match、Token F1、ROUGE-L F1、Citation Precision/Recall/F1。

排序指标作用于最终 `hits`，证据 F1 作用于预算选择后的 `forest`。`Evidence Cost` 是项目内部稳定代理，不是模型服务实际 token。

```text
data/benchmarks/<dataset>/
├── raw/
├── processed/
│   ├── graph.json
│   ├── train.jsonl / dev.jsonl / test.jsonl / all.jsonl
│   ├── base_embeddings.npz
│   └── prepare_report.json
└── reports/
    ├── <split>_<system>.json
    ├── <split>_comparison.csv
    └── <split>_summary.json
```

## 5. 有效性规则

- 下载器校验 ZIP、JSON/JSONL、PDF 文件头和常见图片格式；
- prepare 会校验非空图、非空问题、唯一 query ID、gold/candidate ID 完整性；不会再产出“下载成功但图或 split 为空”的伪成功结果；
- 无法映射的 gold、缺失图片/PDF 会写入 `prepare_report.json`，正式训练和运行均默认拒绝部分或非官方数据；`--allow-partial` 只用于诊断；
- train/dev/test 按论文或文档分组，hard negative、HGT 和调参只使用 train/dev；
- 不完整 gold 会虚高召回，不能作为正式结果；
- 外部 LLM judge、AlignScore 和 MMDocRAG 官方 Judge 尚未集成，需要在固定环境中补测。
