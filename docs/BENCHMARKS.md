# 统一公开数据集评测

所有公开数据集通过一个入口管理：

```bash
paper-rag benchmark <prepare|run|all>
```

当前适配器为 `peerqa` 和 `mmdocrag`。新增数据集只需实现准备函数并注册到 benchmark CLI，不需要复制检索和指标代码。

## 一条命令完成全部流程

```bash
paper-rag benchmark all \
  --datasets peerqa mmdocrag \
  --root data/benchmarks \
  --config configs/default.yaml \
  --setting 20
```

该命令依次执行：

1. 下载官方标注、PDF 和图片；
2. 解压并转换为统一证据图和 JSONL；
3. 为缺失的 PeerQA OpenReview 论文调用 MinerU；
4. 建立向量索引；
5. 顺序运行 BM25、Dense、Reranker、图基线、PCST 和 EC-BFR；
6. 保存每个系统的完整 JSON 报告并生成 CSV/Markdown 对比表。

已存在且非空的下载文件会跳过；中断下载保存在 `.part` 文件并在下次续传。解压和 MinerU 输出也会跳过已完成项。只有显式传入 `--force` 或 `--reindex` 才会重做对应阶段。

正式运行默认检查 `missing_papers`、`missing_images`、`download_errors` 和 `parse_errors`，存在不完整数据时直接停止，防止误把部分数据结果写入论文。只有排查流程时才使用 `--allow-partial`。

MMDocRAG 的 PDF 和图片压缩包较大，应预留足够磁盘空间。若只想先验证文本流程，可以使用 `--skip-pdfs`；图片 quote 仍会下载，因为多模态索引和 Reranker 需要它们。

## 分阶段运行

只下载和转换：

```bash
paper-rag benchmark prepare \
  --datasets peerqa mmdocrag \
  --root data/benchmarks
```

只运行已经准备好的数据：

```bash
paper-rag benchmark run \
  --datasets peerqa mmdocrag \
  --root data/benchmarks \
  --systems bm25 dense dense_reranker one_hop ppr pcst pcst_closure \
            ec_bfr ec_bfr_reranker
```

加入已训练 HGT 的完整方法：

```bash
paper-rag benchmark run \
  --datasets peerqa mmdocrag \
  --systems full \
  --hgt-artifacts outputs/srmg_index
```

默认 `--split official`：PeerQA 使用全部官方可映射问题，MMDocRAG 使用官方 `evaluation_20`。训练实验可显式使用 `--split train`、`--split dev` 或 `--split test`。

## 目录结构

```text
data/benchmarks/
├── peerqa/
│   ├── raw/                 # 官方压缩包、PDF、MinerU 输出
│   ├── processed/           # graph.json、all/train/dev/test.jsonl、索引缓存
│   └── reports/             # 各系统 JSON 和 comparison.csv
└── mmdocrag/
    ├── raw/                 # dev/evaluation JSONL、images.zip、doc_pdfs.zip
    ├── processed/           # quote 图、官方/训练划分、索引缓存
    └── reports/
```

## 两个数据集的处理口径

### PeerQA

官方 `papers.jsonl` 已有的句子直接构图，保留官方 `idx`，因此证据映射不需要重新切句。官方数据中缺少正文的 OpenReview 论文会从 `https://openreview.net/pdf?id=...` 批量下载，再通过 MinerU 补入同一图。无法下载或解析的论文记录在 `prepare_report.json`，不会让整批已完成结果失效。

若机器暂时没有 MinerU，可用 `--skip-mineru` 先准备官方已有文本；这会减少 PeerQA 可评测问题数，不应作为最终论文结果。

### MMDocRAG

默认使用 `setting=20`。每个问题的 text/image quote 直接转换为 Sentence/Figure 节点，并把官方 `gold_quotes` 转成 gold node ID。`candidate_node_ids` 会被传到 BM25 和 Qdrant，因此每个方法都严格在同一组 20 个官方候选中选择，不会误做成全文检索。

`dev_20` 保存为 development，并按 `doc_name` 生成内部 train/dev；`evaluation_20` 原样作为最终 test。开发数据和测试数据的节点 ID 带不同命名空间，即使官方 q_id 重复也不会冲突。

## 输出指标

统一报告包含：

- MRR、Recall@K、nDCG@K、Joint Recall@K；
- Evidence Precision/Recall/F1；
- Sentence/Figure/Caption/ChartData 的排序召回和证据 F1；
- Closure Validity、Dependency Completeness；
- Budget Violation、Evidence Cost、Latency；
- 开启生成时的 Exact Match、Token F1、ROUGE-L 和 Citation F1。

MMDocRAG 的整体 Evidence F1 对应 quote selection F1，Sentence Evidence F1 和 Figure Evidence F1 分别对应文本与图片 quote 选择。BLEU 和官方 LLM Judge 仍应使用 MMDocRAG 官方脚本对生成结果复核。
