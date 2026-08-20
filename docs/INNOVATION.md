# 创新点与实现边界

本文用于区分“已有实现、部分实现、后续实验”，避免把研究设想写成系统现状。算法细节见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 创新点一：科研证据关系监督的多模态图索引

### 已实现

- 使用 Qwen3-VL Embedding 统一编码 query、文本节点和原图；
- 从 MinerU 结果构建 Sentence、Figure、Table、Caption 及其显式关系；
- 图表增强后增加 ChartData 和 `derived_from`；
- 使用两层 HGT 将 2048 维基础表示映射到 256 维结构空间；
- 查询—证据 margin loss，保留问题的全部 gold 证据；
- 优先选择同类型、基础向量相似的 hard negative；
- 对 `caption_of`、`refers_to`、`derived_from`、`next_sentence` 使用关系 InfoNCE；
- 在线阶段缓存节点图向量，只对候选计算 query-HGT 分数；
- 训练产物记录图哈希、query ID 和实际关系三元组数。

### 当前限制

- HGT 对整张图编码，监督按训练论文隔离，属于传导式设置，不是未见论文上的归纳式编码；
- hard negative 只在样本候选或同论文节点中选择，没有跨批次负样本队列；
- MMDocRAG 官方 quote 图通常没有关系边，不能单独验证关系损失；
- 没有实现 R-GCN、GFM-RAG 或 LILaC 的复现代码；
- 当前没有学习式分数校准，基础向量、HGT 和 Reranker 使用 RRF 融合。

### 可验证主张

可以表述为：利用科研论文中的图注、原图、引用句和图表派生数据关系监督轻量异构图适配器，在候选阶段联合建模内容相关性和证据结构。

不能把 HGT、异构图、多模态 Graph RAG 或细粒度文档图本身写成首次提出。

## 2. 创新点二：证据闭包约束的硬预算森林检索

### 已实现

- 按论文扩展候选，并在多个 cost scale 上运行 PCST；
- 在原始有向图上计算最小证据闭包；
- Figure/Table 补 Caption，ChartData 补 Figure，引用图表的 Sentence 补对应节点；
- 闭包后重新计算文本和图片代理成本；
- 淘汰超过总预算的候选；
- 根据相关性、QuerySpec 槽位覆盖、实体新颖性和冗余选择跨论文森林；
- 输出 forest、节点 ID、成本和 PCST 后端元数据；
- 提供 top-k、one-hop、PPR、PCST、PCST+Closure 和 EC-BFR 对比。

### 当前限制

- 槽位覆盖是字符串匹配，不是训练出的语义覆盖模型；
- 实体新颖性依赖 `attributes.entities`，当前 MinerU 流程不自动抽取实体；
- 文本成本是简单 token 代理，图片是固定单位，不是生成模型的真实上下文消耗；
- PCST 候选在论文内生成，跨论文阶段是候选森林选择，不求解统一全局图优化；
- 当前没有 MAGE-RAG Agent 基线。

### 可验证主张

可以表述为：在论文内候选骨架上执行类型化证据闭包，闭包后重新计费，并在严格代理预算内选择跨论文证据森林。

不能把 PCST、预算检索或跨文档图检索本身写成首次提出。

## 3. 图表模块

图表解析是支撑模块，不作为独立核心创新点。当前实现包括人工清单、外部多模态 API、自集成聚合、不确定性记录和 ChartData 来源边。

尚未接入统一 CLI 和 benchmark：自动折线图分类、PP-Chart2Table、DePlot 及专门的图表数据集评测。代码中已有 PP-Chart2Table 和 DePlot 适配器，但当前主流程只使用人工表格或 OpenAI-compatible 服务。

## 4. 必须完成的论文实验

### 创新点一

至少比较：

- Dense；
- Dense + Reranker；
- Full 去掉关系损失；
- Full 去掉 query-evidence loss；
- Full 去掉 hard negative；
- 完整 HGT。

建议增加 R-GCN/HGT 无关系监督基线，以及一个可运行的相关图检索方法。主要指标为 MRR、Recall@K、Joint Recall@K、Evidence F1、分模态召回和延迟。

### 创新点二

使用现有自动矩阵比较 top-k、one-hop、PPR、PCST、PCST+Closure 和 EC-BFR。进一步消融槽位覆盖、实体新颖性、硬预算和跨论文选择。主要指标为 Evidence F1、Closure Validity、Dependency Completeness、Budget Violation、Evidence Cost 和延迟。

当前 CLI 能训练完整 HGT，但尚未提供“关闭单个训练损失”的显式开关。因此关系损失可用 `--relation-weight 0` 消融，其他训练损失和 hard negative 消融仍需补充参数后再形成最终论文表。

## 5. 相关工作边界

- [LILaC](https://github.com/joohyung00/lilac)：粗细粒度组件图和子图检索；
- [MAGE-RAG](https://github.com/laonuo2004/MAGE-RAG)：多粒度页面图和预算化 Agent 导航；
- [GFM-RAG](https://github.com/RManLuo/gfm-rag)：可迁移图检索器；
- [G-Retriever](https://github.com/XiaoxinHe/G-Retriever)：PCST 图检索。

复用的 MinerU、Qwen3-VL、HGTConv、Qdrant 和 `pcst_fast` 都不是项目创新。论文贡献应落在本项目实际实现、能够独立消融并由指标验证的组合机制上。
