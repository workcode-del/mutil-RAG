# 单环境部署与运行

## 1. 部署原则

本项目的默认部署方式是：**一个名为`paper-rag`的Python 3.11 Conda环境、一个代码仓库、一个主配置文件**。

MinerU、PP-Chart2Table、Qwen3-VL Embedding/Reranker、PyG、Qdrant客户端、PCST、API和界面都安装在同一个Conda环境中。模块仍然保持代码隔离，但不再要求切换parser/chart/graph/model等多个环境。

默认在线模式把Embedding和Reranker直接加载到检索API进程。`configs/server.yaml`保留HTTP模式，仅用于显存不足、远程GPU或后期生产部署，不是论文原型的必需步骤。

```text
同一个Conda环境：paper-rag
  ├─ MinerU/PyMuPDF：PDF解析
  ├─ PP-Chart2Table：折线图结构化
  ├─ Qwen3-VL：向量与重排
  ├─ PyG/PCST/Qdrant：图索引与EC-BFR
  └─ FastAPI/Streamlit：系统接口与界面
```

## 2. 平台建议

- Python：3.11；
- GPU：CUDA环境优先，CPU只能用于接口和小数据检查；
- 部署系统：Linux x86-64，推荐Ubuntu 22.04/24.04；
- 生成模型优先使用OpenAI-compatible外部API，避免同一张GPU再常驻一个生成模型。

## 3. 一次性安装

先安装Linux系统库：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl libgl1 libglib2.0-0 libgomp1 fontconfig fonts-noto-cjk
```

在安装了Miniconda或Anaconda的Shell中：

```bash
cd /path/to/mutil-RAG
conda env create -f environment.yml
conda activate paper-rag
```

`environment.yml`使用清华大学`conda-forge`镜像；配置脚本通过`pip config --site`把清华PyPI地址写入当前`paper-rag`环境，不修改用户级或系统级配置：

```text
Conda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
PyPI:  https://pypi.tuna.tsinghua.edu.cn/simple
```

模型权重不经过Conda/PyPI镜像。项目统一采用“本地目录优先、ModelScope下载兜底”，不会默认访问Hugging Face下载权重。下载结果缓存在`data/models`。

先准备Qwen官方代码：

```bash
mkdir -p third_party
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git third_party/Qwen3-VL-Embedding
```

然后执行固定版本安装脚本：

```bash
bash scripts/install_locked.sh
```

脚本按照固定顺序执行：

```text
配置当前Conda环境的清华源
→ 检查torch 2.8 / torchvision 0.23
→ 缺失时安装requirements/torch.txt
→ 安装requirements/locked.txt
→ 用--no-deps安装本项目
→ 用--no-deps安装Qwen官方仓库
→ pip check
```

固定清单位于[requirements/locked.txt](../requirements/locked.txt)。其中PyTorch单独位于[requirements/torch.txt](../requirements/torch.txt)，避免pip在CUDA构建和普通依赖之间反复求解。默认清单使用`torch==2.8.0`和`torchvision==0.23.0`；若部署机需要特定CUDA wheel，可提前安装相同主版本，脚本检测通过后不会覆盖。

MinerU使用`mineru[core]==3.4.4`，不再安装`mineru[all]`带来的vLLM、lmdeploy、S3和Gradio等无关后端。Qwen官方仓库用`--no-deps`安装，避免它再次求解Torch和Transformers。默认配置已经指向`third_party/Qwen3-VL-Embedding`，不需要创建官方仓库自己的uv或Conda环境。

### 3.1 预下载模型

统一依赖中已经包含`modelscope`。安装完成后建议在启动系统前预下载权重：

```bash
paper-rag download-models --config configs/default.yaml \
  --components embedding reranker
```

Qwen模型对应的ModelScope仓库：

- [Qwen3-VL-Embedding-2B](https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B)
- [Qwen3-VL-Reranker-2B](https://modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B)

折线图模型如果确认ModelScope中存在对应仓库，可以追加`chart`：

```bash
paper-rag download-models --config configs/default.yaml \
  --components embedding reranker chart
```

若PP-Chart2Table没有可用的同名ModelScope仓库，应先手动准备完整模型目录，再填写`chart.local_path`。

## 4. 默认本地配置

`configs/default.yaml`的关键配置为：

```yaml
runtime:
  mode: local
  device: cuda
  qwen3_vl_retrieval_repo: third_party/Qwen3-VL-Embedding

model_download:
  source: modelscope
  cache_dir: data/models

embedding:
  backend: qwen3_vl
  model: Qwen/Qwen3-VL-Embedding-2B
  modelscope_id: Qwen/Qwen3-VL-Embedding-2B
  local_path:
  dimension: 2048

reranker:
  enabled: true
  backend: qwen3_vl
  model: Qwen/Qwen3-VL-Reranker-2B
  modelscope_id: Qwen/Qwen3-VL-Reranker-2B
  local_path:

vector_store:
  mode: local
  path: data/index/qdrant
```

该模式不要求启动8101、8102或Qdrant Docker服务。

模型解析顺序固定为：

```text
local_path存在 → 直接返回本地绝对路径
local_path为空或不存在 → 从ModelScope下载到data/models
已经下载过 → ModelScope复用缓存，不重复下载
```

完全离线部署时，把模型目录写入配置并将来源改为`local`：

```yaml
model_download:
  source: local
  cache_dir: data/models

embedding:
  model: Qwen/Qwen3-VL-Embedding-2B
  local_path: /data/models/Qwen3-VL-Embedding-2B

reranker:
  model: Qwen/Qwen3-VL-Reranker-2B
  local_path: /data/models/Qwen3-VL-Reranker-2B

chart:
  model: PaddlePaddle/PP-Chart2Table_safetensors
  local_path: /data/models/PP-Chart2Table_safetensors
```

`source: local`下路径不存在会立即报错，不会访问网络。

## 5. 离线建库

所有命令都在同一个已激活的`paper-rag` Conda环境中执行。每次打开新终端只需先运行`conda activate paper-rag`。

### 5.1 PDF解析

按照[MinerU官方代码](https://github.com/opendatalab/MinerU)解析PDF，得到`content_list.json`和图片目录。随后规范化为证据图：

```bash
paper-rag parse-mineru data/parsed/paper1_content_list.json \
  data/parsed/paper1_graph.json --paper-id paper1
paper-rag inspect-graph data/parsed/paper1_graph.json
```

多篇论文图合并：

```bash
paper-rag merge-graphs data/parsed/evidence_graph.json \
  data/parsed/paper1_graph.json data/parsed/paper2_graph.json
```

当前折线图模块与句级bbox回填仍需进一步接入该建库命令；在接通前不能把`ChartData`和句级定位当作已完成的部署能力。

### 5.2 建立Qdrant索引

脚本会在当前进程直接加载Qwen3-VL Embedding，不再依赖Embedding HTTP服务：

```bash
paper-rag validate-config configs/default.yaml
python scripts/index_graph.py data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --embedding-cache outputs/base_embeddings.npz
```

Qwen3-VL使用2048维向量。旧的1536维索引不能复用。

### 5.3 训练HGT

```bash
python scripts/train_srmg.py \
  data/parsed/evidence_graph.json outputs/base_embeddings.npz \
  --query-samples data/train/query_pairs.jsonl \
  --query-embeddings outputs/query_embeddings.npz \
  --output outputs/srmg_index --epochs 20 --device cuda
```

正式训练必须同时提供查询正负样本和查询Embedding，否则查询投影头没有有效监督。

## 6. 单命令启动在线系统

设置证据图和HGT产物后，在同一个环境用统一入口启动一个API进程：

```bash
paper-rag-serve \
  --graph data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --hgt-artifacts outputs/srmg_index \
  --host 127.0.0.1 --port 8000
```

该进程会依次加载：本地证据图、本地Qdrant、Qwen3-VL Embedding、HGT缓存和Qwen3-VL Reranker。

需要生成答案时增加`--enable-generator`；仅做召回消融时可增加`--disable-reranker`。

健康检查与查询：

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "达到500 MPa强度的材料有哪些？",
    "metric": "strength",
    "value": 500,
    "unit": "MPa"
  }'
```

## 7. 显存不足时的可选方式

“单环境”和“单进程”不是同一件事。如果两套2B模型无法同时进入显存，可以仍然只维护这一个`paper-rag` Conda环境，但在同一环境启动Embedding、Reranker和检索三个进程，并使用`configs/server.yaml`。

这种方式不复制代码、不创建新环境，只隔离模型进程：

```bash
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
