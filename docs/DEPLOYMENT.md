# Linux单环境部署与研究路线

## 1. 运行边界

- 系统：Linux x86-64，推荐Ubuntu 22.04/24.04，Python 3.11；
- 环境：只使用一个名称可自定义的Conda环境，以下用`rag-thesis`示例；
- `paper-rag`和`paper-rag-serve`是程序命令名，不是Conda环境名；
- 主环境固定`transformers==4.57.6`，同时满足MinerU 3.4.4的`<5`约束和Qwen3-VL检索的`>=4.57.3`约束；
- 折线图解析默认调用OpenAI-compatible多模态API。PP-Chart2Table依赖Transformers 5.x，只作为独立服务/对比基线，不装进主环境。

## 2. 安装

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl libgl1 libglib2.0-0 \
  libgomp1 fontconfig fonts-noto-cjk

cd /path/to/mutil-RAG
conda env create --name rag-thesis --file environment.yml
conda activate rag-thesis

mkdir -p third_party
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  third_party/Qwen3-VL-Embedding
bash scripts/install_locked.sh
```

安装脚本只向当前环境写入清华Conda/PyPI源，不允许安装到`base`。模型优先读取`configs/default.yaml`中的`local_path`；路径为空时从ModelScope下载：

```bash
paper-rag download-models --config configs/default.yaml \
  --components embedding reranker
```

若之前遇到Transformers冲突，更新代码后在已激活环境中重新执行：

```bash
python -m pip install --upgrade "transformers==4.57.6"
bash scripts/install_locked.sh
python -m pip check
```

## 3. 从PDF到系统的完整衔接

| 步骤 | 需要的输入/组件 | 命令 | 生成物 | 交给下一步 |
|---|---|---|---|---|
| 1. PDF解析 | `data/pdfs/*.pdf`；MinerU | `mineru` | `*_content_list.json`和图片 | 构图 |
| 2. 细粒度构图 | MinerU JSON | `paper-rag parse-mineru` | 单篇证据图JSON | 折线图增强 |
| 3. 折线图增强 | 图JSON、折线图清单；多模态API或人工CSV | `list-figures`、`enrich-charts` | 含ChartData的图JSON | 多论文合并 |
| 4. 多论文合并 | 多个增强图 | `merge-graphs` | `evidence_graph.json` | 向量索引/HGT |
| 5. 多模态基础索引 | 合并图、Qwen3-VL Embedding | `index_graph.py` | Qdrant目录、2048维NPZ | HGT训练/在线召回 |
| 6. 结构化索引训练 | 查询正负证据标注、2048维NPZ | `embed_training_queries.py`、`train_srmg.py` | 256维HGT图索引 | 在线结构增强 |
| 7. 检索与回答 | 图、Qdrant、HGT、Reranker | `paper-rag-serve` | 最小证据森林、回答和证据ID | 系统展示/评测 |
| 8. 实验评测 | 查询及`relevant_node_ids` | `evaluate_retrieval.py` | Precision/Recall/F1和预算违规率 | 论文实验表 |

### 3.1 PDF解析

输入：论文PDF。外部组件：[MinerU](https://github.com/opendatalab/MinerU)。国内下载模型时使用ModelScope：

```bash
mkdir -p data/pdfs data/mineru data/parsed
export MINERU_MODEL_SOURCE=modelscope
mineru -p data/pdfs -o data/mineru -b pipeline
find data/mineru -name '*_content_list.json'
```

对每篇论文执行，输入路径替换为`find`显示的真实文件：

```bash
paper-rag parse-mineru data/mineru/paper1/paper1_content_list.json \
  data/parsed/paper1_graph.json --paper-id paper1 \
  --pdf data/pdfs/paper1.pdf
paper-rag inspect-graph data/parsed/paper1_graph.json
```

`--pdf`让PyMuPDF把MinerU块级位置细化为句级bbox；命令输出中的`sentence_locations`是成功定位数量。最终图包含Sentence、Figure、Caption节点以及`caption_of`、`refers_to`等边，并保留页码、bbox和图片路径。这是创新点一的异构证据图原料。

### 3.2 只处理折线图

先导出全部图片节点：

```bash
paper-rag list-figures data/parsed/paper1_graph.json \
  data/parsed/paper1_line_charts.jsonl
```

人工查看`image_path`，删除非折线图行。保留行至少包含：

```json
{"figure_id":"paper1:figure:12"}
```

自动解析需要一个支持图片输入的OpenAI-compatible接口。设置`configs/default.yaml`中的`chart.base_url`和`chart.model`，需要密钥时执行：

```bash
export PAPER_RAG_API_KEY='你的密钥'
paper-rag enrich-charts data/parsed/paper1_graph.json \
  data/parsed/paper1_line_charts.jsonl \
  data/parsed/paper1_enriched.json \
  --config configs/default.yaml
```

程序对每张图重复解析3次，按单元格聚合并记录不确定性，生成ChartData节点和`derived_from`边。没有API时，可在清单中人工填写`linearized_table`，程序将直接入图：

```json
{"figure_id":"paper1:figure:12","linearized_table":"strain,stress_MPa\n0.01,420\n0.02,510","confidence":1.0}
```

### 3.3 合并论文并建立多模态索引

```bash
paper-rag merge-graphs data/parsed/evidence_graph.json \
  data/parsed/paper1_enriched.json data/parsed/paper2_enriched.json

paper-rag validate-config configs/default.yaml
python scripts/index_graph.py data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --embedding-cache outputs/base_embeddings.npz
```

输出：`data/index/qdrant`保存分类向量索引，`outputs/base_embeddings.npz`保存所有句子、图、图注和ChartData的2048维向量。前者用于在线召回，后者用于HGT训练。

### 3.4 训练创新点一：结构监督HGT索引

准备`data/train/query_pairs.jsonl`。正负节点ID必须存在于`evidence_graph.json`：

```json
{"query_id":"q1","query":"达到500 MPa强度的材料有哪些？","positive_node_id":"paper1:sentence:8:1","negative_node_id":"paper1:sentence:3:0"}
```

公开数据可从带证据标注的QA样本转换；私有数据可人工标注少量查询。只有答案、没有证据位置的数据不能直接训练或评测细粒度检索。

```bash
python scripts/embed_training_queries.py data/train/query_pairs.jsonl \
  outputs/query_embeddings.npz --config configs/default.yaml

python scripts/train_srmg.py \
  data/parsed/evidence_graph.json outputs/base_embeddings.npz \
  --query-samples data/train/query_pairs.jsonl \
  --query-embeddings outputs/query_embeddings.npz \
  --output outputs/srmg_index --epochs 20 --device cuda
```

输出`graph_embeddings.npy`、`node_ids.json`和`query_projector.pt`。它们把Figure-Caption-Mention-ChartData关系变成结构监督分数，实现创新点一“多模态结构化索引”。

### 3.5 运行创新点二和完整系统

```bash
paper-rag-serve \
  --graph data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --hgt-artifacts outputs/srmg_index \
  --host 127.0.0.1 --port 8000
```

查询：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"达到500 MPa强度的材料有哪些？",
    "metric":"strength",
    "value":500,
    "unit":"MPa"
  }'
```

在线流程为：分类向量召回 → HGT结构分数 → Qwen3-VL原图重排 → PCST候选骨架 → 证据闭包 → 重算文本/图片成本 → 预算内选择跨论文证据森林。返回的`forest`、`total_cost`和证据ID直接体现创新点二“证据闭包与预算约束检索”。

如需自然语言回答，先确保`generation.base_url`指向可用的OpenAI-compatible多模态模型服务，再增加`--enable-generator`。不启用时系统仍完整返回可评测的证据森林。

### 3.6 公开数据和私有数据评测

把SciGraphQA、SciVQA或自建样本统一成JSONL，并把原始证据位置映射为本系统node_id：

```json
{"query_id":"test-1","query":"Which material exceeds 500 MPa?","relevant_node_ids":["paper1:sentence:8:1","paper1:figure:12"]}
```

```bash
python scripts/evaluate_retrieval.py data/eval/public.jsonl \
  --graph data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --hgt-artifacts outputs/srmg_index \
  --output outputs/public_metrics.json

python scripts/evaluate_retrieval.py data/eval/private.jsonl \
  --graph data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --hgt-artifacts outputs/srmg_index \
  --output outputs/private_metrics.json
```

消融实验：去掉`--hgt-artifacts`得到“无结构索引”；增加`--disable-reranker`得到“无多模态重排”。两者均与完整系统使用同一份图和评测数据。

## 4. 外部组件清单

| 组件 | 是否必须 | 用途 |
|---|---|---|
| MinerU | 必须 | PDF转结构化块、图片、页码和bbox |
| Qwen3-VL-Embedding/Reranker官方仓库 | 必须 | 多模态召回与原图重排 |
| ModelScope | 本地无模型时必须 | 下载并缓存模型权重 |
| OpenAI-compatible多模态API | 折线图自动解析、生成答案时需要 | 图转CSV和最终答案；可用云API或独立模型服务 |
| PP-Chart2Table | 非必须、仅基线 | 依赖Transformers 5.x，应独立部署，不能与MinerU主环境混装 |

## 5. 当前验收标准

1. `pip check`无冲突，`transformers`版本为4.57.6；
2. 合并图中node_id唯一，边端点存在，折线图具有ChartData→Figure来源边；
3. Qdrant和基础NPZ均为2048维，HGT产物为256维；
4. 训练样本的正负节点ID和评测样本的证据ID均存在于同一版图中；
5. 查询结果`total_cost`不超过配置预算，回答引用ID必须属于返回森林。

当前设备不能运行模型时只能完成静态检查；GPU模型、MinerU解析质量和API连通性必须在Linux部署机上实测。
