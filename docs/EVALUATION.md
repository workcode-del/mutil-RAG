# 自定义数据评测

PeerQA 和 MMDocRAG 使用自动入口，见 [BENCHMARKS.md](BENCHMARKS.md)。本文只说明自定义图和 JSONL 的单次评测，不重复公开数据下载流程。

## 1. 数据格式

每行一个问题：

```json
{
  "query_id":"q1",
  "paper_id":"paper1",
  "query":"What is the main contribution?",
  "answer":"reference answer",
  "relevant_node_ids":["paper1:sentence:8:1"],
  "candidate_node_ids":["paper1:sentence:8:1","paper1:sentence:9:0"]
}
```

字段约定：

| 字段 | 必需 | 含义 |
|---|---:|---|
| `query_id` | 否 | 问题 ID；缺省时使用行号 |
| `query` | 是 | 查询文本 |
| `relevant_node_ids` | 是 | gold 证据 ID，至少一个 |
| `paper_id` 或 `paper_ids` | 否 | `sample` 范围下限制检索论文 |
| `candidate_node_ids` | 否 | 限制候选集合；必须包含全部 gold |
| `answer` | 否 | 启用生成时计算答案指标 |
| `required_modalities` | 否 | 问题需要的证据模态 |

评测前应确保所有证据和候选 ID 都存在于同一版图中。自定义证据是文本时，可用 `scripts/prepare_peerqa.py` 的精确/模糊匹配逻辑作为转换工具，但它不是公开 PeerQA 主入口。

## 2. 建立索引

BM25 不需要索引；Dense、Reranker 和 HGT 需要先执行：

```bash
paper-rag index data/parsed/evidence_graph.json \
  --embedding-cache data/cache/base_embeddings.npz \
  --config configs/default.yaml
```

## 3. 单次实验

BM25：

```bash
python scripts/evaluate_retrieval.py data/eval/test.jsonl \
  --graph data/parsed/evidence_graph.json \
  --candidate-backend bm25 \
  --retrieval-method top_k \
  --disable-reranker \
  --output outputs/eval/bm25.json
```

完整检索方法：

```bash
python scripts/evaluate_retrieval.py data/eval/test.jsonl \
  --graph data/parsed/evidence_graph.json \
  --candidate-backend embedding \
  --retrieval-method ec_bfr \
  --hgt-artifacts outputs/hgt \
  --ranking-k 1 3 5 10 \
  --output outputs/eval/full.json
```

`--scope sample` 按样本的 `paper_id/paper_ids` 限制检索，是默认值；`--scope corpus` 用于跨论文实验。启用生成需增加 `--enable-generator` 并配置外部模型服务。

## 4. 对比矩阵

在相同图、问题、候选范围、top-k 和预算下运行：

| 方法 | candidate backend | retrieval method | 其他开关 |
|---|---|---|---|
| BM25 | `bm25` | `top_k` | 关闭 Reranker |
| Dense | `embedding` | `top_k` | 关闭 Reranker |
| Dense + Reranker | `embedding` | `top_k` | 默认 Reranker |
| One-hop | `embedding` | `one_hop` | — |
| PPR | `embedding` | `ppr` | — |
| PCST | `embedding` | `pcst` | — |
| PCST + Closure | `embedding` | `pcst_closure` | — |
| EC-BFR | `embedding` | `ec_bfr` | 不传 HGT |
| Full | `embedding` | `ec_bfr` | 传入 HGT |

汇总多个报告：

```bash
python scripts/compare_evaluations.py outputs/eval/*.json \
  --output outputs/eval/comparison.csv
```

## 5. 当前指标

- 排序：MRR、MRR@10、Recall@K、nDCG@K、Joint Recall@K；
- 最终证据集：Evidence Precision、Recall、F1；
- 节点类型：Sentence、Figure、Caption、ChartData 的召回与证据 F1；
- 结构：Closure Validity、Dependency Completeness；
- 预算与效率：Budget Violation、Evidence Cost、Selected Nodes、Latency；
- 可选生成：Exact Match、Token F1、ROUGE-L F1、Citation Precision/Recall/F1。

排序指标作用于融合和图重排后的 `hits`，证据 F1 作用于最终 `forest`。`Evidence Cost` 是项目内部代理成本，不是模型服务实际 token 用量。

## 6. 实验规范

- 按 `paper_id` 划分 train/dev/test，避免同一论文的问题跨集合；
- hard negative、HGT 和超参数只使用训练集或开发集；
- 测试集只用于最终报告；
- 不完整 gold 会虚高召回，不能默认允许部分证据映射；
- 除目标消融项外，其余模型、图、候选范围、预算和 top-k 保持一致；
- 外部 LLM judge、AlignScore 和 MMDocRAG 官方 Judge 当前未集成，应在固定的独立环境中补测。
