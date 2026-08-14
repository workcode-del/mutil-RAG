# PeerQA 与 MMDocRAG 自动评测

公开数据统一使用：

```bash
paper-rag benchmark <prepare|train|run|all>
```

## 1. 一键运行

准备数据、建立索引并比较无需训练的方法：

```bash
paper-rag benchmark all \
  --datasets peerqa mmdocrag \
  --root data/benchmarks \
  --config configs/default.yaml \
  --setting 20
```

同时训练 HGT，并把 `full` 加入 held-out test 对比：

```bash
paper-rag benchmark all \
  --datasets peerqa mmdocrag \
  --root data/benchmarks \
  --config configs/default.yaml \
  --setting 20 \
  --train-hgt
```

`all` 会依次下载和转换数据、建立 Dense 索引、可选训练 HGT、运行系统矩阵并生成 JSON 报告和 `comparison.csv`。已有下载、解压和 MinerU 结果会复用；`--force` 重做数据准备，`--reindex` 重建向量索引。

评测 query 默认按 `--query-batch-size 64` 批量编码，并在使用同一 embedding 模型的系统间复用。排名与指标仍逐样本计算；报告记录批量/单条编码的最小余弦一致性，低于 `0.999` 时终止。`latency_mode=batch_amortized_end_to_end` 表示延迟包含摊销后的批量编码时间，不等同于在线单请求延迟。

## 2. 分阶段运行

```bash
# 下载和转换
paper-rag benchmark prepare \
  --datasets peerqa mmdocrag --root data/benchmarks

# 训练每个数据集自己的 HGT
paper-rag benchmark train \
  --datasets peerqa mmdocrag --root data/benchmarks \
  --config configs/default.yaml

# 运行已准备的数据
paper-rag benchmark run \
  --datasets peerqa mmdocrag --root data/benchmarks \
  --systems bm25 dense dense_reranker one_hop ppr pcst \
            pcst_closure ec_bfr ec_bfr_reranker

# 评测训练后的完整方法
paper-rag benchmark run \
  --datasets peerqa mmdocrag --root data/benchmarks \
  --split test --systems full \
  --hgt-artifacts outputs/benchmark_hgt
```

HGT 产物默认写入 `outputs/benchmark_hgt/<dataset>`。运行时检查图哈希以及训练 query 与评测 query 是否重叠。

## 3. 数据处理口径

### PeerQA

- 从 PeerQA 官方 Hugging Face 仓库下载固定版本的 `qa.jsonl` 和 `papers.jsonl`；
- 直接使用 `papers.jsonl` 中的官方句子和 `idx` 构图；
- 默认评测官方可再分发文本子集；`--peerqa-download-pdfs` 可尝试下载 OpenReview PDF，并用 MinerU 扩展图；
- 按 `paper_id` 稳定划分 train/dev/test；
- `--split official` 使用全部成功映射的问题。

官方可再分发文本子集不依赖额外 PDF，直接运行统一命令即可。扩展实验使用：

```bash
paper-rag benchmark all --datasets peerqa --peerqa-download-pdfs
```

扩展结果应单独报告成功下载和解析的论文范围。`--allow-partial` 只用于排错。

### MMDocRAG

- 从官方 Hugging Face 仓库下载 `dev_<setting>.jsonl`、`evaluation_<setting>.jsonl` 和 `images.zip`；
- 默认 `setting=20`，即每个问题在官方 20 个 text/image quote 中选择证据；
- quote 转为 `Sentence` 或 `Figure`，以问题内候选 `quote_id` 建立节点；
- `gold_quotes` 转为 gold node ID，`candidate_node_ids` 限制 BM25 和 Dense 使用完全相同的候选；
- `dev_20` 按文档划为内部 train/dev，`evaluation_20` 作为 test；
- `--split official` 等同于 test。

官方候选协议不需要 PDF，默认不下载 `doc_pdfs.zip`。`--mmdocrag-download-pdfs` 目前只下载并解压 PDF，尚未自动把全文 MinerU 图与 quote 标注对齐。

MMDocRAG quote 图不虚构 `next_sentence` 等关系，因此 HGT 的 `training.json` 可能显示 `relation_triples: 0`。此时只训练了 query-evidence 目标，不能作为关系监督有效性的主证据。全文关系图实验应单独实现标注对齐，并明确报告为 full-document protocol。

## 4. 默认系统

默认运行：

- `bm25`
- `dense`
- `dense_reranker`
- `one_hop`
- `ppr`
- `pcst`
- `pcst_closure`
- `ec_bfr`
- `ec_bfr_reranker`

`full` 需要 HGT 产物，不在无训练默认列表中。各系统的具体差异见 [EVALUATION.md](EVALUATION.md)。

## 5. 输出

```text
data/benchmarks/<dataset>/
├── raw/                 # 下载文件、图片、PDF、MinerU 输出
├── processed/
│   ├── graph.json
│   ├── train.jsonl / dev.jsonl / test.jsonl
│   ├── base_embeddings.npz
│   ├── dense_index.json
│   └── prepare_report.json
└── reports/
    ├── <split>_<system>.json
    ├── <split>_comparison.csv
    └── <split>_summary.json
```

报告包含排序、证据集、节点类型、闭包、预算和延迟指标；启用生成时增加答案和引用指标。MMDocRAG 的 Evidence F1 可用于 quote selection 对比，但 BLEU 和官方 LLM Judge 尚未集成，需用其官方评测工具补充。
