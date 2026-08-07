# 创新点、最新相关工作与论文表述边界

## 1. 2025–2026相关工作带来的边界

以下表述不能再使用：首次多模态Graph RAG、首次细粒度分层文档图、首次多模态子图检索、首次在预算内选择证据、首次使用PCST。

- [LILaC（EMNLP 2025 Main）](https://arxiv.org/abs/2602.04263)已经构建粗细粒度组件图并用晚交互与Beam Search检索子图；[代码](https://github.com/joohyung00/lilac)。
- [MAGE-RAG（2026预印本）](https://arxiv.org/abs/2606.15906)已经构建多粒度页面证据图并进行显式预算的Agent导航；[代码](https://github.com/laonuo2004/MAGE-RAG)。
- [MG²-RAG（2026预印本）](https://arxiv.org/abs/2604.04969)已经研究多粒度多模态图与相关性传播。
- [GFM-RAG（NeurIPS 2025）/G-reasoner（ICLR 2026）](https://github.com/RManLuo/gfm-rag)提供可迁移的预训练图检索器。
- [G-Retriever](https://github.com/XiaoxinHe/G-Retriever)和[BRIT](https://aclanthology.org/2025.findings-emnlp.1211/)已经使用PCST检索相关子图。

因此，本论文贡献必须限制在“科研证据关系监督”“证据依赖闭包”“闭包后硬预算”和“句子/原图/曲线数据可定位”这些组合明确、可实现和可验证的性质上。最终是否能写“首次”，仍需在论文定稿前做系统文献检索；当前文档不做未经验证的唯一性承诺。

## 2. 创新点一：图结构约束的科研多模态索引

### 方法

1. 使用[Qwen3-VL-Embedding-2B](https://arxiv.org/abs/2601.04720)统一编码查询、句子、图注、ChartData和原始Figure；
2. 从PDF结构自动获得Figure-Caption、Figure-Mention、Figure-ChartData关系；
3. 使用关系InfoNCE将真实结构边作为正样本，同论文相似但错误配对作为难负样本；
4. 用两层HGT和查询投影头学习2048→256维结构空间；
5. 在线阶段由Qdrant全库召回，HGT只增强候选，随后由Qwen3-VL-Reranker直接读取原图和文本精排；
6. 返回Sentence、Figure、Caption和ChartData节点及其page/bbox，而非停留在页面或段落。

### 与LILaC的区别

LILaC的主要方法是通用分层组件图、模态查询分解和晚交互遍历；本方法不把“分层图”作为新颖性，而研究科研论文显式证据关系如何监督统一向量空间和轻量图适配器。首版避免依赖72B查询分解模型，便于硕士阶段实现。

### 建议论文表述

> 提出一种图结构约束的科研多模态异构索引方法，利用论文中Figure-Caption-Mention-ChartData显式关系构造跨模态训练信号，并通过轻量异构图适配器联合建模内容语义和证据结构，实现原句、原图和曲线数据级定位。

## 3. 创新点二：证据依赖闭包约束的硬预算森林检索

### 方法

1. 使用相关性作为节点prize、关系类型作为边cost，由PCST生成多尺度候选骨架；
2. 定义科研证据依赖规则及最小不动点闭包；
3. 在原始有向图上补全必要证据，而不是把任意邻居全部加入；
4. 闭包后重新计算文本token与图片成本，超预算候选直接淘汰；
5. 按问题槽位覆盖、实体新颖性、相关性和冗余选择多个论文分量；
6. 输出最小、完整、可引用的跨论文证据森林。

### 与MAGE-RAG的区别

MAGE-RAG通过Agent执行Activate/Open/Search/Prune操作并受过程预算控制。本方法是确定性图优化流程，不训练或调用Agent选择动作；重点研究Figure、Caption、Mention和ChartData的必要依赖、闭包后的硬预算可行性以及跨论文多答案证据组合。

### 建议论文表述

> 提出一种证据依赖闭包约束的预算森林检索方法，在论文内候选骨架上补齐科研解释所必需的类型化依赖，闭包后重新计算上下文代价，并在严格预算下选择覆盖问题槽位且保持答案多样性的跨论文证据森林。

## 4. 必须完成的实验

### 创新点一

- Qwen3-VL-Embedding flat；
- Embedding + Qwen3-VL-Reranker；
- R-GCN/HGT无关系监督；
- 完整关系监督HGT；
- LILaC或其可运行简化基线；
- 可选GFM-RAG文本化图基线。

消融：去除ChartData、去除难负样本、去除关系损失、去除查询排序损失、图片仅用caption而不输入原图。

指标：Sentence Recall、Figure Recall、Joint Evidence Recall、MRR、Evidence F1、定位成功率和延迟。

### 创新点二

- top-k；
- top-k+一跳；
- PPR；
- 普通PCST；
- PCST+闭包；
- 完整EC-BFR；
- 若资源允许，增加MAGE-RAG或固定预算Agent基线。

消融：去闭包、去槽位覆盖、去实体新颖性、软预算替代硬预算、单论文替代跨论文森林。

指标：Closure Validity、Slot Coverage、Evidence F1、Citation Precision、预算违反率、平均证据成本和QA质量-预算曲线。

## 5. 图表模块的研究边界

PP-Chart2Table和VLM自集成是系统支撑模块，不建议占用两个核心创新点之一。采用[Self-Ensembling Vision-Language Models for Chart Data Extraction](https://arxiv.org/abs/2605.27298)的重复解析、中位数聚合和不确定性思想；[DePlot](https://arxiv.org/abs/2212.10505)仅作为传统基线。实验可使用[Chart-MRAG Bench/CHARGE](https://arxiv.org/abs/2502.14864)及其[代码](https://github.com/Nomothings/CHARGE)。

## 6. 开源复用边界

| 开源组件 | 直接复用 | 本项目新增 |
|---|---|---|
| MinerU | PDF布局、图片、bbox | 统一证据契约、句级定位、关系构图 |
| Qwen3-VL-Embedding | 多模态基础表示 | 科研关系监督与图索引适配 |
| Qwen3-VL-Reranker | 原图/文本相关性 | 图节点载荷、分数融合、消融 |
| PP-Chart2Table | 图到表 | 来源链、不确定性、闭包依赖 |
| PyG HGTConv | 异构消息传递 | 图模式、训练样本、双空间检索 |
| pcst_fast | PCST求解 | 证据闭包、闭包后预算、跨论文森林 |
| Qdrant | ANN与payload过滤 | 类型平衡召回和证据ID映射 |

严谨性原则：复用的模型不是创新；只有拥有独立公式、实现、消融和中间指标的模块才写入创新点。
