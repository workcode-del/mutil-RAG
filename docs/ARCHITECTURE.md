# 系统架构

本文只描述当前代码已经实现的主流程。运行命令见 [DEPLOYMENT.md](DEPLOYMENT.md)，公开数据评测见 [BENCHMARKS.md](BENCHMARKS.md)。

## 1. 主流程

```text
PDF
 └─ MinerU → Sentence / Figure / Caption 证据图
      └─ 可选图表增强 → ChartData
           └─ Qwen3-VL Embedding → Qdrant + 基础向量缓存
                └─ 可选 HGT 候选增强
                     └─ 可选 Qwen3-VL Reranker
                          └─ PCST 候选 → 证据闭包 → 硬预算森林
                               └─ 可选 OpenAI-compatible 生成
```

在线阶段使用 RRF 融合基础向量、HGT 和 Reranker 的排序。HGT 只给已召回节点打分，不替代 Qdrant 全库检索。

## 2. 证据图

当前 MinerU 适配器实际生成：

| 节点 | 内容 |
|---|---|
| `Sentence` | 切分后的正文句子 |
| `Figure` | MinerU 导出的原图路径 |
| `Caption` | 独立或图片内嵌图注 |
| `ChartData` | 图表增强得到的线性化表格 |

当前自动构建的关系：

- `Sentence --next_sentence--> Sentence`
- `Caption --caption_of--> Figure`
- `Sentence --refers_to--> Figure`
- `ChartData --derived_from--> Figure`

节点保留 `node_id`、`paper_id`、页码、bbox、解析块 ID 和来源信息。`Paper`、`Section`、`Paragraph`、`contains`、`semantically_similar` 已在数据结构中预留，但当前 MinerU 主流程不生成，不能作为现有实验能力报告。

MMDocRAG 官方协议直接把候选 quote 转成 `Sentence` 或 `Figure`。该图不推断原数据没有提供的关系，详见 [BENCHMARKS.md](BENCHMARKS.md)。

## 3. 基础索引与 HGT

基础索引对文本节点和原图分别编码为 2048 维向量，写入 Qdrant，并保存同一份 NPZ 供训练使用。检索时按节点类型分别取 top-k，减少文本节点数量对图片召回的挤压。

HGT 使用配置中的节点类型投影和两层异构消息传递：

```text
node:  2048 → type projection → HGT → 256
query: 2048 → MLP                  → 256
```

训练包含两类目标：

1. 查询—证据 margin loss：每个 gold 节点配一个同类型 hard negative；
2. 关系 InfoNCE：使用 `caption_of`、`refers_to`、`derived_from` 和 `next_sentence` 边。

训练关系只取训练论文，但模型对整张图计算节点表示。因此当前实现属于按论文隔离监督的传导式图编码；最终报告必须保证训练 query 与测试 query 不重叠。训练产物中的 `training.json` 记录图哈希、训练 query ID 和关系三元组数。

## 4. 检索器

代码提供以下统一检索接口：

| 方法 | 实现 |
|---|---|
| `top_k` | 按融合分数选前 k 个节点 |
| `one_hop` | top-k 后扩展一跳邻居 |
| `ppr` | 在已召回节点子图上做 PPR 重排 |
| `pcst` | 每篇论文选择一个 PCST 候选 |
| `pcst_closure` | PCST 后补全证据依赖 |
| `ec_bfr` | 多尺度 PCST、闭包后计费、跨论文硬预算选择 |

EC-BFR 的闭包规则以最小不动点执行：选中 Figure 补 Caption，选中 ChartData 补 Figure，选中引用图的 Sentence 补 Figure；迭代后也会补齐该 Figure 的 Caption。

成本模型是稳定代理值：文本按简单 token 规则估算，图片按固定 `image_unit` 计费。它用于方法内公平比较，不等同于生成模型的真实计费 token。槽位覆盖读取 `QuerySpec` 中的字段；实体新颖性依赖节点已有的 `attributes.entities`，当前 MinerU 流程没有自动 NER，因此该项通常不生效。

## 5. 图表与生成

`list-figures` 导出全部 Figure，当前没有自动折线图分类器，需要人工筛选清单。`enrich-charts` 可读取人工提供的 `linearized_table`，也可调用 OpenAI-compatible 多模态服务重复解析并聚合，生成 `ChartData --derived_from--> Figure`。

生成模块是可选项。它把证据文本和原图发送到 OpenAI-compatible `/chat/completions`，要求返回 JSON：

```json
{"answer":"...","evidence_ids":["paper:sentence:1:0"]}
```

程序只接受当前证据森林中的 ID，不能验证每个自然语言断言是否真正由对应证据支持。

## 6. 代码边界

| 模块 | 职责 |
|---|---|
| `parsing`、`evidence_graph` | MinerU 适配、句级定位、构图和图表增强 |
| `embedding`、`reranking` | 多模态召回、BM25 和重排 |
| `models`、`training.py` | HGT、训练损失和离线产物 |
| `retrieval` | top-k、PPR、PCST、闭包和 EC-BFR |
| `evaluation`、`benchmarking` | 指标、公开数据转换和实验矩阵 |
| `workflow.py`、`bootstrap.py` | 批处理流程和运行时组件装配 |
