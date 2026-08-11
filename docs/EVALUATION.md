# RAG 评测与对比实验

这套评测把“候选召回”和“证据森林选择”分开计分。这样可以判断提升来自向量模型、重排器、图扩展，还是 EC-BFR 的闭包与预算约束，而不是只看最终回答的主观质量。

PeerQA 与 MMDocRAG 的批量下载、转换、索引、实验矩阵和汇总现在统一使用 `paper-rag benchmark`；完整命令见 [统一公开数据集评测](BENCHMARKS.md)。下面保留的是底层格式和单次实验入口，便于调试自定义数据。

## 1. 数据集

首选 [PeerQA](https://github.com/UKPLab/PeerQA)：它包含 208 篇论文、579 个真实同行评审问题，并提供作者标注的答案与句子级证据。官方检索主指标是 MRR 和 Recall@10，正好适合本项目的细粒度证据检索。

PeerQA 因论文许可限制不直接分发所有 PDF。先按其官方 README 下载 `qa.jsonl` 和各来源 PDF，再用 MinerU 解析 PDF。解析时建议直接使用 PeerQA 的 `paper_id`：

```bash
paper-rag parse-mineru <content_list.json> data/parsed/<paper_id>.json \
  --paper-id <paper_id> --pdf <paper.pdf>
paper-rag merge-graphs data/parsed/peerqa_graph.json data/parsed/*.json
```

若本地图使用了短 ID，建立映射文件：

```json
{
  "openreview:ICLR-2022-conf:xxxx": "local-paper-id"
}
```

将 PeerQA 的证据文本映射到 MinerU 图节点：

```bash
python scripts/prepare_peerqa.py data/peerqa/qa.jsonl \
  --graph data/parsed/peerqa_graph.json \
  --paper-id-map data/peerqa/paper_id_map.json \
  --output data/evaluation/peerqa.jsonl \
  --report outputs/peerqa_mapping.json
```

转换器先做 Unicode/空白归一化和精确匹配，再做模糊匹配。默认只写入全部证据均成功映射的问题；`outputs/peerqa_mapping.json` 保留逐条分数，必须人工抽查低分匹配。不要为了增加样本数而默认使用 `--allow-partial`，因为残缺 gold 会虚高 Recall。

通用评测 JSONL 也可以手工构造，每行格式为：

```json
{"query_id":"q1","paper_id":"dflash","query":"What is the main contribution?","answer":"...","relevant_node_ids":["dflash:sentence:67:0"]}
```

## 2. 一次建索引

BM25 基线直接读取图，不需要索引。向量方法先执行：

```bash
python scripts/index_graph.py data/parsed/peerqa_graph.json --config configs/default.yaml
```

所有方法必须使用同一份图、同一批问题、相同 `--per-type-top-k`、`--selection-top-k` 和检索范围。PeerQA 属于单论文问答，默认 `--scope sample` 只在该问题对应论文内检索，符合官方设置；跨论文实验才使用 `--scope corpus`。

## 3. 基线与消融

先运行无需模型的最小基线：

```bash
python scripts/evaluate_retrieval.py data/evaluation/peerqa.jsonl \
  --graph data/parsed/peerqa_graph.json --candidate-backend bm25 \
  --retrieval-method top_k --disable-reranker \
  --output outputs/eval/bm25_top_k.json
```

再按同一配置依次运行以下实验。`dense` 表示 `--candidate-backend embedding`：

| 实验 | 关键参数 | 作用 |
|---|---|---|
| Dense Top-k | `--retrieval-method top_k --disable-reranker` | 向量召回基线 |
| Dense + Reranker | `--retrieval-method top_k` | 重排器贡献 |
| + One-hop | `--retrieval-method one_hop` | 简单图扩展基线 |
| + PPR | `--retrieval-method ppr` | 图传播基线 |
| + PCST | `--retrieval-method pcst` | 候选骨架贡献 |
| + PCST + Closure | `--retrieval-method pcst_closure` | 证据依赖闭包贡献 |
| EC-BFR | `--retrieval-method ec_bfr` | 完整硬预算方法 |
| EC-BFR + HGT | 上项再加 `--hgt-artifacts <dir>` | 学习式图增强贡献 |

例如完整方法：

```bash
python scripts/evaluate_retrieval.py data/evaluation/peerqa.jsonl \
  --graph data/parsed/peerqa_graph.json --candidate-backend embedding \
  --retrieval-method ec_bfr --hgt-artifacts outputs/hgt \
  --ranking-k 1 3 5 10 --output outputs/eval/ec_bfr_hgt.json
```

启用回答生成和外部 API 时，先配置 `generation.api_key_env`，再导出同名环境变量，并添加 `--enable-generator`。例如默认配置为：

```bash
export PAPER_RAG_API_KEY='<your-api-key>'
```

## 4. 指标

| 维度 | 指标 | 含义 |
|---|---|---|
| 排序 | MRR、MRR@10、Recall@K、nDCG@K、Joint Recall@K | 首个证据、全部证据与排序质量 |
| 证据集 | Evidence Precision/Recall/F1 | 最终选择的森林是否命中 gold |
| 多模态 | Sentence/Figure/Caption/ChartData Recall | 各节点类型的召回能力 |
| 结构 | Closure Validity、Dependency Completeness | 图注、原图、引用句等依赖是否完整 |
| 预算 | Budget Violation、Evidence Cost、Selected Nodes | 是否严格遵守上下文预算及成本 |
| 效率 | Latency | 单问题端到端检索时间 |
| 生成 | Exact Match、Token F1、ROUGE-L F1 | 有参考答案且启用生成时计算 |
| 引用 | Citation Precision/Recall/F1 | 生成答案引用的节点是否属于 gold |

PeerQA 原论文的检索结果可直接对齐 MRR 和 Recall@10；MRR@10 是额外的截断指标。答案生成原论文还使用 AlignScore/模型裁判；当前代码不内置这类重量级或模型依赖指标，因此生成对比应报告 ROUGE-L，并在独立、固定版本的评测环境中补充 AlignScore 或统一 LLM judge。

## 5. 生成对比表

```bash
python scripts/compare_evaluations.py outputs/eval/*.json \
  --output outputs/evaluation_comparison.csv
```

命令会在终端打印 Markdown 表，同时保存 CSV。论文主表至少报告 MRR、Recall@10、Evidence F1、Closure Validity、Budget Violation、Evidence Cost 和 Latency；消融实验除目标组件外必须保持参数一致。

为避免数据泄漏，阈值、预算和超参数只能在开发集确定，测试集只运行一次最终配置。若数据没有官方划分，应按 `paper_id` 划分而不是按问题随机划分，防止同一论文同时进入开发集和测试集。
