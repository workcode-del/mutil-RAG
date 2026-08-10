# 系统架构、模块契约与算法联系

## 1. 运行边界

| 环境/服务 | 主要依赖 | 输入 | 输出 |
|---|---|---|---|
| parser | 当前MinerU（论文依据MinerU2.5）、PyMuPDF | PDF | 规范证据图JSON、Figure图片、page/bbox |
| chart | PP-Chart2Table或外部VLM | 仅折线图图片 | ChartData、置信度、不确定性、来源 |
| embedding | Qwen3-VL-Embedding-2B | query、文本、图片 | 2048维归一化向量 |
| reranker | Qwen3-VL-Reranker-2B | query与文本/原图/混合证据 | 相关性分数 |
| graph | PyG、Qdrant、pcst_fast | 证据图、缓存向量、query | 候选节点、闭包证据森林 |
| generation | Qwen3-VL兼容API | 证据森林与图片 | 带合法evidence_id的回答 |

默认部署在同一个名为`paper-rag`的Python 3.11 Conda环境中，Embedding、Reranker和图检索由一个API进程直接组装，便于论文阶段统一修改和调试。模块接口仍然保留HTTP实现；只有显存不足、使用远程GPU或后期生产部署时，才在同一Conda环境中把模型拆成多个进程。

## 2. 数据图谱

### 节点

- 粗粒度：Paper、Section、Paragraph、Figure；
- 细粒度：Sentence、Caption、ChartData；
- 后续扩展：ChartSeries、Axis、Legend。首版可以把序列、坐标轴和图例保存在ChartData属性中，避免过早扩大图模式。

### 显式关系

- `Paper/Section/Paragraph --contains--> 下级节点`；
- `Caption --caption_of--> Figure`；
- `Sentence --refers_to--> Figure`；
- `ChartData --derived_from--> Figure`；
- `Sentence --next_sentence--> Sentence`；
- `Node --semantically_similar--> Node`，仅作为低置信度弱边。

每个节点保存稳定`node_id`、`paper_id`、`page`、`bbox`和provenance。图表派生节点额外保存extractor、parse_status、confidence和uncertainty。

## 3. 结构化多模态索引（创新点一）

### 3.1 基础召回

```text
Sentence/Caption/ChartData --Qwen3-VL text--> R^2048
Figure                    --Qwen3-VL image-> R^2048
Query                     --Qwen3-VL query-> R^2048
```

向量进入Qdrant，并按节点类型分别取top-k，防止大量句子淹没图片。Qwen3-VL-Embedding支持原图、文本和混合输入；当前索引对文本与原图独立编码，以保持索引简单，混合编码作为消融项。

### 3.2 结构增强

基础向量通过节点类型投影和两层HGT：

```text
R^2048 --W_type--> R^256 --HGT(typed edges)--> graph_embedding R^256
Query R^2048 --Wq/MLP------------------------> query_graph R^256
```

训练信号来自PDF真实结构：Figure-Caption、Figure-Mention、Figure-ChartData为正关系，同论文相似但不对应的图片或文字作为难负样本。HGT只重排Qdrant候选，不承担全库ANN，因此可以独立关闭并做消融。

HGT发表于2020年，但这里只是基础消息传递算子；论文方法依据和新颖性来自科研证据关系监督，而不是声称提出新的HGT。

### 3.3 原图多模态精排

Qwen3-VL-Reranker直接接收：

```json
{"query": {"text": "..."}, "documents": [{"text": "..."}, {"image": "fig.png", "text": "caption..."}]}
```

这替代旧版“Figure只转成caption文本后重排”的限制。Qdrant余弦、HGT分数和Reranker相关性默认用RRF融合，避免直接相加不同量纲；正式实验可在验证集上训练校准器。

## 4. EC-BFR检索（创新点二）

1. 将召回结果按paper_id分组，并扩展高置信度显式邻居；
2. 对多个代价尺度运行PCST，生成若干论文内候选骨架；
3. 回到原始有向异构图计算证据闭包，而不是在PCST无向图上解释依赖；
4. Figure补Caption，ChartData补原Figure和Caption；需要时补引用句与实验条件；
5. 闭包后去重、重新计费，超预算候选直接淘汰；
6. 按相关性、问题槽位覆盖、实体新颖性和冗余惩罚选择跨论文森林。

必须满足：

- 幂等性：`C(C(S)) = C(S)`；
- 来源完整：ChartData必须能回链原图和图注；
- 硬预算：`Cost(Forest) <= Budget`；
- 跨论文分量无需伪造连边。

PCST是候选生成器而非创新本身。MAGE-RAG已经研究预算化多模态图导航；本项目区别是确定性类型闭包、闭包后重新计费、严格预算和跨论文森林，不依赖LLM Agent在线试错。

## 5. 图表解析

主后端为PP-Chart2Table；也可把Qwen3-VL或外部API包装成相同接口。训练无关的自集成层对同一图重复解析3次，对齐表格后对数值单元格取中位数，并把结果分歧转成uncertainty。低置信度数值可以参与召回，但回答精确数值时必须同时展示原图并标注不确定性。

DePlot保留为2023传统基线，不再是主后端。只处理折线图，架构图、流程图和显微图不进入ChartData解析。

## 6. 回答与可追溯性

`serialize_forest`把证据写成`[evidence_id] type/page/content`，原图路径单独传给Qwen3-VL。生成结果必须返回answer和evidence_ids；程序校验ID必须属于当前证据森林，阻止模型伪造来源。界面再由ID查回paper_id、page、bbox和image_path。

## 7. 消融开关

| 模式 | HGT关系监督 | 闭包/森林 | 图像精排 | 用途 |
|---|---:|---:|---:|---|
| Qwen3-VL flat | 关 | 关 | 关 | 基础召回 |
| Embedding + VL-Reranker | 关 | 关 | 开 | 强平坦基线 |
| LILaC式分层候选基线 | 关 | 关 | 开 | 2025强相关基线 |
| SRMG-Index | 开 | 关 | 开 | 创新点一 |
| PCST | 可选 | 关 | 开 | 普通相关子图 |
| EC-BFR | 可选 | 开 | 开 | 创新点二 |
| Full | 开 | 开 | 开 | 最终系统 |

参考依据：[Qwen3-VL检索](https://arxiv.org/abs/2601.04720)、[LILaC](https://arxiv.org/abs/2602.04263)、[MAGE-RAG](https://arxiv.org/abs/2606.15906)、[GFM-RAG代码](https://github.com/RManLuo/gfm-rag)。
