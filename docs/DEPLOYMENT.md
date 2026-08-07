# 运行、部署与迁移文档

当前设备只能看代码时无需执行本文件。正式运行推荐Windows主机+WSL2 Ubuntu 24.04+NVIDIA GPU。先用10篇PDF完成验收，再扩大数据集。

## 1. 重要迁移说明

旧版GME向量为1536维，新版Qwen3-VL-Embedding-2B默认为2048维，不能在同一Qdrant collection中混用。升级时必须：

1. 保留原始解析JSON和图片；
2. 新建或清空目标Qdrant collection；
3. 使用Qwen3-VL重新生成全部节点向量；
4. 重新训练HGT并导出2048→256维查询投影头；
5. 不复用旧`graph_embeddings.npy`或`query_projector.pt`。

## 2. 前置条件

- Python 3.11或3.12；
- Git、Docker Desktop、WSL2；
- NVIDIA驱动与WSL CUDA；
- `uv`或conda；
- 建议至少16GB内存；模型服务显存以目标机器smoke test为准；
- Qwen3-VL Embedding和Reranker均为2B，显存不足时顺序启动并离线缓存，不要求同时常驻。

## 3. 主项目环境

```bash
cd "/mnt/d/GitHub project/mutil RAG"
python -m venv .venv-graph
source .venv-graph/bin/activate
pip install -e ".[dev,api,app,vector,graph]"
```

PyG和CUDA版本必须按[PyTorch Geometric安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)选择对应wheel。Windows上`pcst_fast`编译失败时使用WSL；fallback仅用于界面演示，正式PCST实验必须记录真实求解器后端。

## 4. PDF解析环境

```bash
python -m venv .venv-parser
source .venv-parser/bin/activate
pip install -r requirements/parser.txt
pip install -e .
```

按照[MinerU官方代码](https://github.com/opendatalab/MinerU)下载模型并解析PDF。论文依据更新为[MinerU2.5](https://arxiv.org/abs/2509.22186)，适配层读取MinerU输出JSON：

```bash
paper-rag parse-mineru data/parsed/paper1_content_list.json \
  data/parsed/paper1_graph.json --paper-id paper1
paper-rag inspect-graph data/parsed/paper1_graph.json
```

每篇论文检查句子bbox、Figure图片、Caption和引用句。只有折线图进入ChartData解析。

## 5. Qwen3-VL检索模型环境

官方实现：[Qwen3-VL-Embedding代码](https://github.com/QwenLM/Qwen3-VL-Embedding)。本项目不复制其源码，服务通过环境变量定位官方仓库。

```bash
mkdir -p third_party
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git third_party/Qwen3-VL-Embedding
cd third_party/Qwen3-VL-Embedding
uv sync
source .venv/bin/activate
uv pip install -e "/mnt/d/GitHub project/mutil RAG"

export QWEN3_VL_RETRIEVAL_REPO="$PWD"
export EMBEDDING_MODEL="Qwen/Qwen3-VL-Embedding-2B"
export EMBEDDING_DIMENSION=2048
export EMBEDDING_DEVICE=cuda
uvicorn services.embedding_api:app --app-dir "/mnt/d/GitHub project/mutil RAG" \
  --host 127.0.0.1 --port 8101
```

健康检查：

```bash
curl http://127.0.0.1:8101/health
```

另一个进程启动多模态重排器：

```bash
export QWEN3_VL_RETRIEVAL_REPO="/absolute/path/to/third_party/Qwen3-VL-Embedding"
export RERANKER_MODEL="Qwen/Qwen3-VL-Reranker-2B"
export RERANKER_DEVICE=cuda
uvicorn services.reranker_api:app --app-dir "/mnt/d/GitHub project/mutil RAG" \
  --host 127.0.0.1 --port 8102
```

官方仓库也支持vLLM>=0.14.0；首版建议先用Transformers适配器，减少服务层变量。

## 6. 折线图解析环境

```bash
python -m venv .venv-chart
source .venv-chart/bin/activate
pip install -r requirements/chart.txt
pip install -e .
```

主后端为`PaddlePaddle/PP-Chart2Table_safetensors`，调用代码位于`src/paper_rag/chart/pp_chart2table.py`。自集成包装器默认重复3次，数值取中位数并输出不确定性。若本机无法运行该模型，可将Qwen3-VL或外部多模态API包装为同一`extract(image_path)`接口；DePlot仅用于对比实验。

## 7. Qdrant与索引构建

本地原型可用Qdrant Local；服务模式：

```bash
docker compose -f deploy/docker-compose.yml up -d qdrant
```

在Embedding服务健康后建立2048维索引：

```bash
paper-rag validate-config configs/default.yaml
python scripts/index_graph.py data/parsed/evidence_graph.json \
  --config configs/default.yaml --embedding-cache outputs/base_embeddings.npz
```

不要把旧1536维collection改名后继续使用；必须重新向量化。

## 8. 训练轻量结构适配器

```bash
source .venv-graph/bin/activate
python scripts/train_srmg.py \
  data/parsed/evidence_graph.json outputs/base_embeddings.npz \
  --query-samples data/train/query_pairs.jsonl \
  --query-embeddings outputs/query_embeddings.npz \
  --output outputs/srmg_index --epochs 20 --device cuda
```

训练脚本从NPZ自动推断输入维度。输出包括`graph_embeddings.npy`、`node_ids.json`和`query_projector.pt`。关系监督样本和query-evidence样本都没有时训练会拒绝启动，避免导出未训练索引。

## 9. 回答生成服务

采用[Qwen3-VL官方实现](https://github.com/QwenLM/Qwen3-VL)，推荐4B或8B；显存不足时使用2B或OpenAI-compatible外部API。配置默认为：

```yaml
generation:
  provider: openai_compatible
  base_url: http://127.0.0.1:8001/v1
  model: Qwen/Qwen3-VL-4B-Instruct
```

外部服务需支持图片URL或base64。当前生成器会保留image_path，部署适配层必须将本地路径转换为服务可访问的载荷，不能假装已经上传。

## 10. 启动检索API与界面

```bash
export PAPER_RAG_GRAPH=data/parsed/evidence_graph.json
export PAPER_RAG_CONFIG=configs/server.yaml
export PAPER_RAG_HGT_ARTIFACTS=outputs/srmg_index
export PAPER_RAG_ENABLE_RERANKER=1
export PAPER_RAG_ENABLE_GENERATOR=0
uvicorn services.retrieval_api:app --host 0.0.0.0 --port 8000
```

生成服务验证后把`PAPER_RAG_ENABLE_GENERATOR`改为1。启动界面：

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

## 11. 无模型静态检查

```bash
pip install -e ".[dev]"
paper-rag validate-config configs/default.yaml
pytest
python -m compileall -q src services scripts app tests
```

这些检查验证接口、schema、闭包、预算和Python语法，不代表GPU模型已经运行。

## 12. 十篇论文验收门

1. Gate A：抽查30个句子bbox和20组Figure-Caption-Mention；
2. Gate B：Embedding/Qdrant均为2048维，HGT输入2048维、输出256维且无NaN；
3. Gate C：原图能进入VL-Reranker，而非只输入caption；
4. Gate D：图表自集成结果保存extractor、confidence、uncertainty和原图来源；
5. Gate E：闭包幂等，所有森林严格不超预算；
6. Gate F：20个问题返回evidence_id、页码、原句/原图，生成引用不越界。

六项全部通过后再冻结环境、记录GPU与峰值显存，并开展公开集和私有集实验。
