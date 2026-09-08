# 系统架构

本文只描述当前代码已经实现的主流程。运行命令见 [DEPLOYMENT.md](DEPLOYMENT.md)，评测协议见 [EVALUATION.md](EVALUATION.md)。

## 1. 主流程

```text
PDF
 └─ MinerU → Sentence / Figure / Table / Caption 证据图
      └─ 确定性科研实体抽取 → 类型、规范名、置信度与来源
      └─ 可选图表增强 → ChartData
           └─ Qwen3-VL Embedding → Qdrant + 基础向量缓存
                └─ 可选 HGT 候选增强 / R-GCN 对照
                     └─ 可选 Qwen3-VL Reranker
                          └─ PCST 候选 → 证据闭包 → 硬预算森林
                               └─ 可选 OpenAI-compatible 生成
```

查询进入检索前先自动补全 `QuerySpec` 的答案类型、实体类型、指标、比较符、数值、单位、条件、模态和科研实体；调用方显式给出的字段优先，不会被覆盖。在线阶段使用 RRF 融合基础向量、图模型和 Reranker 的排序。HGT/R-GCN 只给已召回节点打分，不替代 Qdrant 全库检索；多模态 Reranker 只处理配置的前 `top_n` 个候选。

## 2. 证据图

当前 MinerU 适配器实际生成：

| 节点 | 内容 |
|---|---|
| `Sentence` | 切分后的正文句子 |
| `Figure` | MinerU 导出的原图路径 |
| `Table` | MinerU 表格 HTML/文本及可选的表格截图 |
| `Caption` | 独立或图、表内嵌的图注/表注 |
| `ChartData` | 图表增强得到的线性化表格 |

当前自动构建的关系：

- `Sentence --next_sentence--> Sentence`
- `Caption --caption_of--> Figure`
- `Caption --caption_of--> Table`
- `Sentence --refers_to--> Figure/Table`
- `ChartData --derived_from--> Figure`

节点保留 `node_id`、`paper_id`、页码、bbox、解析块 ID 和来源信息。`Paper`、`Section`、`Paragraph`、`contains`、`semantically_similar` 已在数据结构中预留，但当前 MinerU 主流程不生成，不能作为现有实验能力报告。

MMDocRAG 官方协议直接把候选 quote 转成 `Sentence` 或 `Figure`。该图不推断原数据没有提供的关系，详见 [EVALUATION.md](EVALUATION.md)。

## 3. 基础索引与图模型

基础索引把文本节点、原图和表格分别编码为 2048 维向量，写入 Qdrant，并保存同一份 NPZ。图训练首次使用该缓存时生成连续矩阵 sidecar，之后一次装入 GPU；batch 只传递节点行号和边索引。Table 有截图时使用图文联合编码，只有 HTML/文本时使用文本编码。检索与重排均显式包含 Table，并按节点类型分别取 top-k，减少文本数量对图片和表格召回的挤压。

HGT 使用配置中的节点类型投影和两层异构消息传递：

```text
node:  2048 → type projection → HGT → 256
query: 2048 → MLP                  → 256
```

R-GCN 强基线复用完全相同的节点/查询投影、训练样本、hard negative、关系边、损失权重、层数和输出维度，只把异构消息传递器替换为按类型化边三元组编号的关系卷积。因而 `rgcn` 与 `full` 的差异只在图编码器，适合作为 SRMG 的关键结构对照。

训练包含两类目标：

1. 查询—证据 margin loss：每个 gold 节点配一个同类型 hard negative；
2. 关系 InfoNCE：使用 `caption_of`、`refers_to`、`derived_from` 和 `next_sentence` 边。

训练 batch 先按论文或跨论文集合聚合同一批监督，再把多个论文组装到不超过 `batch_size` 的 step；只收集当前 batch 论文的节点特征和关系边。验证/测试论文的节点与边不参与梯度，训练完成后才用共享类型投影和消息传递参数逐论文导出全图表示。因此该实现是在固定节点/关系类型模式下对未见论文做归纳编码，不应扩大为对未见节点类型或关系类型的泛化。A800 默认采用 BF16 混合精度，归一化、损失、梯度裁剪、模型主参数和优化器状态保持 FP32。训练产物中的 `training.json` 记录图哈希、训练 query ID、关系三元组数、batch 策略和实际精度。

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

EC-BFR 的闭包规则以最小不动点执行：只沿显式标记为强制且置信度达标的依赖边，选中 Figure 或 Table 补 Caption，选中 ChartData 补 Figure，选中引用图表的 Sentence 补对应 Figure/Table；迭代后继续补齐依赖。Table 成本包含线性化文本，并在有截图时额外计入一个图片单位。多尺度 λ 产生同一论文的候选替代方案，森林最多保留每篇论文的一棵树。

成本模型是稳定代理值：文本按简单 token 规则估算，图片按固定 `image_unit` 计费。它用于方法内公平比较，不等同于生成模型的真实计费 token。槽位覆盖读取自动补全后的 `QuerySpec`，并验证指标、带单位数值、比较关系、实体类型和条件是否由候选证据满足。构图保存或旧图加载时会自动抽取材料、模型、方法、数据集、化学式和命名科研术语，保存规范名、类型、位置、置信度及抽取版本；EC-BFR 据查询实体类型计算有界实体新颖性。当前抽取器是可复现的高精度规则系统，不等同于通用学习式 NER。

## 5. 图表与生成

`list-figures` 导出全部 Figure，当前没有自动折线图分类器，需要人工筛选清单。`enrich-charts` 可读取人工提供的 `linearized_table`，调用 OpenAI-compatible 多模态服务重复解析并聚合，或使用 PP-Chart2Table/DePlot 本地后端，生成 `ChartData --derived_from--> Figure`。空、可疑或低置信度结果会被跳过。

生成模块是可选项。它把正文、表格文本和 Figure/Table 截图作为标准多模态
`messages` 发送到 OpenAI-compatible `/chat/completions`，并从
`choices[0].message.content` 读取普通文本答案。该标准字段为空时，适配器兼容读取部分
供应商使用的 `reasoning_content` 或 `reasoning` 字段；`content` 始终优先，避免把思考
过程覆盖最终答案。供应商扩展请求参数可选地通过 `generation.extra_body` 透传。通用
对话接口不提供项目内部证据 ID，因此该后端不计算引用指标；检索阶段的
`selected_node_ids` 仍完整记录生成上下文的来源。

## 6. 代码边界

| 模块 | 职责 |
|---|---|
| `parsing`、`evidence_graph` | MinerU 适配、句级定位、构图和图表增强 |
| `embedding`、`reranking` | 多模态召回、BM25 和重排 |
| `query_understanding.py`、`entities.py` | QuerySpec 自动解析和可审计科研实体抽取 |
| `models`、`training.py` | HGT、R-GCN、共享训练损失和离线产物 |
| `retrieval` | top-k、PPR、PCST、闭包和 EC-BFR |
| `evaluation`、`benchmarking` | 指标、公开数据转换和实验矩阵 |
| `workflow.py`、`bootstrap.py` | 批处理流程和运行时组件装配 |
