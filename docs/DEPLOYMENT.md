# 单环境部署与运行

## 1. 部署原则

本项目的默认部署方式是：**一个Python 3.11虚拟环境、一个代码仓库、一个主配置文件**。

MinerU、PP-Chart2Table、Qwen3-VL Embedding/Reranker、PyG、Qdrant客户端、PCST、API和界面都安装在同一个虚拟环境中。模块仍然保持代码隔离，但不再要求切换parser/chart/graph/model等多个环境。

默认在线模式把Embedding和Reranker直接加载到检索API进程。`configs/server.yaml`保留HTTP模式，仅用于显存不足、远程GPU或后期生产部署，不是论文原型的必需步骤。

```text
同一个.venv
  ├─ MinerU/PyMuPDF：PDF解析
  ├─ PP-Chart2Table：折线图结构化
  ├─ Qwen3-VL：向量与重排
  ├─ PyG/PCST/Qdrant：图索引与EC-BFR
  └─ FastAPI/Streamlit：系统接口与界面
```

## 2. 平台建议

- Python：3.11；
- GPU：CUDA环境优先，CPU只能用于接口和小数据检查；
- Windows下如果`pcst-fast`或MinerU安装失败，使用WSL2 Ubuntu，但仍只创建一个虚拟环境；
- 生成模型优先使用OpenAI-compatible外部API，避免同一张GPU再常驻一个生成模型。

## 3. 一次性安装

PowerShell：

```powershell
cd "D:\GitHub project\mutil RAG"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

CUDA机器先按照[PyTorch官方安装说明](https://pytorch.org/get-started/locally/)在当前`.venv`安装匹配CUDA的PyTorch，然后安装统一依赖：

```powershell
pip install -e ".[unified]"
```

也可以使用等价入口：

```powershell
pip install -r requirements/all.txt
```

Qwen3-VL Embedding与Reranker共用一个官方仓库，也安装到当前`.venv`：

```powershell
New-Item -ItemType Directory -Force third_party
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git third_party/Qwen3-VL-Embedding
pip install -e third_party/Qwen3-VL-Embedding
```

默认配置已经指向`third_party/Qwen3-VL-Embedding`，不需要再创建官方仓库自己的uv环境。

## 4. 默认本地配置

`configs/default.yaml`的关键配置为：

```yaml
runtime:
  mode: local
  device: cuda
  qwen3_vl_retrieval_repo: third_party/Qwen3-VL-Embedding

embedding:
  backend: qwen3_vl
  model: Qwen/Qwen3-VL-Embedding-2B
  dimension: 2048

reranker:
  enabled: true
  backend: qwen3_vl
  model: Qwen/Qwen3-VL-Reranker-2B

vector_store:
  mode: local
  path: data/index/qdrant
```

该模式不要求启动8101、8102或Qdrant Docker服务。

## 5. 离线建库

所有命令都在同一个已激活的`.venv`中执行。

### 5.1 PDF解析

按照[MinerU官方代码](https://github.com/opendatalab/MinerU)解析PDF，得到`content_list.json`和图片目录。随后规范化为证据图：

```powershell
paper-rag parse-mineru data/parsed/paper1_content_list.json `
  data/parsed/paper1_graph.json --paper-id paper1
paper-rag inspect-graph data/parsed/paper1_graph.json
```

多篇论文图合并：

```powershell
paper-rag merge-graphs data/parsed/evidence_graph.json `
  data/parsed/paper1_graph.json data/parsed/paper2_graph.json
```

当前折线图模块与句级bbox回填仍需进一步接入该建库命令；在接通前不能把`ChartData`和句级定位当作已完成的部署能力。

### 5.2 建立Qdrant索引

脚本会在当前进程直接加载Qwen3-VL Embedding，不再依赖Embedding HTTP服务：

```powershell
paper-rag validate-config configs/default.yaml
python scripts/index_graph.py data/parsed/evidence_graph.json `
  --config configs/default.yaml `
  --embedding-cache outputs/base_embeddings.npz
```

Qwen3-VL使用2048维向量。旧的1536维索引不能复用。

### 5.3 训练HGT

```powershell
python scripts/train_srmg.py `
  data/parsed/evidence_graph.json outputs/base_embeddings.npz `
  --query-samples data/train/query_pairs.jsonl `
  --query-embeddings outputs/query_embeddings.npz `
  --output outputs/srmg_index --epochs 20 --device cuda
```

正式训练必须同时提供查询正负样本和查询Embedding，否则查询投影头没有有效监督。

## 6. 单命令启动在线系统

设置证据图和HGT产物后，在同一个环境用统一入口启动一个API进程：

```powershell
paper-rag-serve `
  --graph data/parsed/evidence_graph.json `
  --config configs/default.yaml `
  --hgt-artifacts outputs/srmg_index `
  --host 127.0.0.1 --port 8000
```

该进程会依次加载：本地证据图、本地Qdrant、Qwen3-VL Embedding、HGT缓存和Qwen3-VL Reranker。

需要生成答案时增加`--enable-generator`；仅做召回消融时可增加`--disable-reranker`。

健康检查与查询：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
  query = "达到500 MPa强度的材料有哪些？"
  metric = "strength"
  value = 500
  unit = "MPa"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" -Body $body
```

## 7. 显存不足时的可选方式

“单环境”和“单进程”不是同一件事。如果两套2B模型无法同时进入显存，可以仍然只维护这一个`.venv`，但在同一环境启动Embedding、Reranker和检索三个进程，并使用`configs/server.yaml`。

这种方式不复制代码、不创建新环境，只隔离模型进程：

```powershell
python -m uvicorn services.embedding_api:app --port 8101
python -m uvicorn services.reranker_api:app --port 8102
python -m uvicorn services.retrieval_api:app --port 8000
```

## 8. 静态验收门

当前设备不能运行模型时，至少检查：

1. `configs/default.yaml`为本地backend；
2. Embedding/Qdrant/HGT输入均为2048维，HGT输出为256维；
3. 图中node_id唯一，边的端点都存在；
4. HGT训练提供query samples和query embeddings；
5. PCST正式实验记录`pcst_fast`后端，而不是fallback；
6. 闭包后森林成本不超过预算；
7. 生成器引用ID必须属于当前证据森林。

这些检查只能验证代码契约，不等同于GPU模型已经成功运行。
