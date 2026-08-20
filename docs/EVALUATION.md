# 统一评测

公开 benchmark 与自定义证据图共用同一套检索系统、指标和报告格式。部署与建库见 [DEPLOYMENT.md](DEPLOYMENT.md)，算法定义见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 公开数据集

一条命令完成数据准备、Dense 索引、系统矩阵和汇总；`--train-hgt` 还会按隔离划分训练 HGT，并加入 `full`：

```bash
paper-rag benchmark all \
  --datasets peerqa mmdocrag \
  --root data/benchmarks \
  --config configs/default.yaml \
  --setting 20 \
  --train-hgt
```

也可分阶段运行：

```bash
paper-rag benchmark prepare --datasets peerqa mmdocrag --root data/benchmarks
paper-rag benchmark train --datasets peerqa mmdocrag --root data/benchmarks \
  --config configs/default.yaml
paper-rag benchmark run --datasets peerqa mmdocrag --root data/benchmarks \
  --config configs/default.yaml --split test --systems dense full \
  --hgt-artifacts outputs/benchmark_hgt
```

已有下载和解析结果会复用；`--force` 重做准备，`--reindex` 重建 embedding 缓存。缓存 sidecar 校验图哈希和 embedding 配置哈希，模型或 query instruction 改变时自动失效。benchmark 在 NPZ 上做精确 cosine 检索，不依赖 Qdrant；在线服务仍使用配置的向量库。

### 数据集与口径

| 数据集 | 检索单位与范围 | 默认 Recall@K | 数据来源 |
|---|---|---|---|
| PeerQA | 官方句子；按论文限制候选 | 1/3/5/10 | 官方 Hugging Face 文本 |
| MMDocRAG | 每题官方 text/image quotes | 1/3/5/10/15/20 | 官方 Hugging Face dev/evaluation |
| M3DocVQA | open-domain PDF 页面 | 1/3/5/10 | 用户提供官方/LILaC 快照 |
| MMLongBench-Doc | 单文档页面，支持多页 gold | 1/3/5/10 | 官方问题与 PDF |
| MultimodalQA | open-domain text/table/image 组件 | 1/3/5/10 | LILaC 指定的官方快照 |

显式 `--ranking-k` 覆盖默认值。`comparison.csv` 导出整体与 Sentence/Figure/Table 分模态 Recall、Evidence F1、结构、预算和延迟指标。

#### PeerQA 与 MMDocRAG

- PeerQA 直接使用官方 `papers.jsonl` 的句子和 `idx` 构图，按 `paper_id` 稳定划分；`--peerqa-download-pdfs` 仅用于 MinerU 全文扩展实验。
- MMDocRAG 默认 `setting=20`，把 quote 转成 Sentence/Figure，并用 `candidate_node_ids` 保证所有方法使用相同候选；dev 用于内部 train/dev，evaluation 为 test。
- MMDocRAG quote 图不虚构全文关系，因此 `relation_triples` 可能为 0。下载 PDF 的 `--mmdocrag-download-pdfs` 目前不负责把全文节点与 quote gold 对齐。

#### 新增的跨论文多模态数据

MMLongBench-Doc 自动下载 `samples.json` 和实际使用的 PDF，只保留带 `evidence_pages` 的问题。官方 gold 页号从 1 开始，节点沿用相同页码：

```bash
paper-rag benchmark prepare --datasets mmlongbench_doc --root data/benchmarks
```

`--max-documents 5` 只用于冒烟测试，报告会标记为 `partial_documents`。已有官方快照可通过 `--dataset-source "mmlongbench_doc=/data/MMLongBench-Doc"` 使用。

MultimodalQA 默认下载 `JoohyungYun/multimodalqa_doc`，读取 `QAs_dev_labeled.json`、`parsed_documents/dev` 和 `image_components/dev`。text、table、image 分别建成 Sentence、Table、Figure：

```bash
paper-rag benchmark prepare --datasets multimodalqa --root data/benchmarks
```

M3DocVQA 官方语料需要动态构造，因此不静默使用第三方镜像。输入快照需包含 `M3DocVQA_dev_labeled.json` 与 `pdf_pages/dev`：

```bash
paper-rag benchmark prepare --datasets m3docvqa \
  --dataset-source "m3docvqa=/data/M3DocVQA"
```

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

`query` 和非空 `relevant_node_ids` 必需。`paper_id/paper_ids` 限制论文范围，`candidate_node_ids` 进一步限制候选且必须包含全部 gold；`answer` 只在启用生成时使用。

建立索引并运行单次实验：

```bash
paper-rag index data/parsed/evidence_graph.json \
  --embedding-cache data/cache/base_embeddings.npz \
  --config configs/default.yaml

python scripts/evaluate_retrieval.py data/eval/test.jsonl \
  --graph data/parsed/evidence_graph.json \
  --candidate-backend embedding \
  --retrieval-method ec_bfr \
  --hgt-artifacts outputs/hgt \
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
- 无法映射的 gold、缺失图片/PDF 会写入 `prepare_report.json`，正式运行默认拒绝部分数据；`--allow-partial` 只用于诊断；
- train/dev/test 按论文或文档分组，hard negative、HGT 和调参只使用 train/dev；
- 不完整 gold 会虚高召回，不能作为正式结果；
- 外部 LLM judge、AlignScore 和 MMDocRAG 官方 Judge 尚未集成，需要在固定环境中补测。
