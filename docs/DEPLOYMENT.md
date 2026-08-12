# 部署与运行

## 1. 环境安装

目标环境为 Linux x86-64、Python 3.11 和 CUDA。仓库使用一个 Conda 环境；`paper-rag` 是安装后的命令名，不是环境名。

```bash
cd /path/to/mutil-RAG
conda env create --name rag-thesis --file environment.yml
conda activate rag-thesis

mkdir -p third_party
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git \
  third_party/Qwen3-VL-Embedding
bash scripts/install_locked.sh
python -m pip check
```

依赖版本以 `requirements/locked.txt` 和 `requirements/torch.txt` 为准。主环境固定 Transformers 4.x；依赖 Transformers 5.x 的 PP-Chart2Table 不在主环境安装。

## 2. 配置与模型

主要配置位于 `configs/default.yaml`：

- `embedding`、`reranker`：本地模型、ModelScope ID 和设备；
- `vector_store`：Qdrant 本地目录或服务地址；
- `graph_index`：HGT 维度和层数；
- `retrieval`：候选扩展、PCST 尺度和预算；
- `chart`、`generation`：外部 OpenAI-compatible 服务。

预下载模型：

```bash
paper-rag download-models --config configs/default.yaml \
  --components embedding reranker
```

默认下载到 `model_download.cache_dir`，当前配置为 `data/models`。若配置了存在的 `local_path`，程序直接使用本地目录。

外部服务需要密钥时，配置 `api_key_env` 并导出同名变量：

```bash
export PAPER_RAG_API_KEY='<api-key>'
```

## 3. 批量构建语料

把 PDF 放入 `data/pdfs`，一条命令完成 MinerU 解析、合图、基础向量缓存和 Qdrant 索引：

```bash
export MINERU_MODEL_SOURCE=modelscope
paper-rag build-corpus data/pdfs \
  --mineru-output data/mineru \
  --graph data/parsed/evidence_graph.json \
  --embedding-cache data/cache/base_embeddings.npz \
  --config configs/default.yaml
```

命令跳过已有 MinerU 结果；`--force` 重新解析全部 PDF。当前增量判断按 PDF 文件名匹配 MinerU content list，因此同名 PDF 不应放在不同子目录。

需要检查单篇解析结果时使用：

```bash
paper-rag parse-mineru '<content_list.json>' data/parsed/paper1.json \
  --paper-id paper1 --pdf data/pdfs/paper1.pdf
paper-rag inspect-graph data/parsed/paper1.json
```

`--pdf` 使用 PyMuPDF 尝试把块级 bbox 细化到句子位置；定位失败不会删除句子节点。

## 4. 可选图表增强

当前流程不自动识别折线图。先导出 Figure 清单，人工删除非目标图片：

```bash
paper-rag list-figures data/parsed/evidence_graph.json \
  data/parsed/line_charts.jsonl
```

每行保留 `figure_id`。可以直接提供表格：

```json
{"figure_id":"paper1:figure:12","linearized_table":"x,y\n1,2","confidence":1.0}
```

也可以只提供 `figure_id`，由 `chart.base_url` 指向的多模态服务解析：

```bash
paper-rag enrich-charts data/parsed/evidence_graph.json \
  data/parsed/line_charts.jsonl \
  data/parsed/evidence_graph_chart.json \
  --config configs/default.yaml

paper-rag index data/parsed/evidence_graph_chart.json \
  --embedding-cache data/cache/base_embeddings_chart.npz \
  --config configs/default.yaml
```

## 5. 训练 HGT

训练 JSONL 必须包含 `query_id`、`query`、`paper_id` 和 `relevant_node_ids`，可选 `candidate_node_ids`。所有节点 ID 必须属于同一版图。

```bash
paper-rag train-index \
  --graph data/parsed/evidence_graph.json \
  --samples data/train/train.jsonl \
  --base-embeddings data/cache/base_embeddings.npz \
  --output outputs/hgt \
  --config configs/default.yaml \
  --epochs 20 --device cuda
```

输出：

- `graph_embeddings.npy`：256 维节点表示；
- `node_ids.json`：矩阵行与证据 ID 的映射；
- `query_projector.pt`：在线 query 投影；
- `training.json`：图哈希、训练 query 和关系三元组统计。

公开数据的准备、训练和测试可直接使用 `paper-rag benchmark all --train-hgt`，见 [BENCHMARKS.md](BENCHMARKS.md)。

## 6. 启动服务

```bash
paper-rag-serve \
  --graph data/parsed/evidence_graph.json \
  --config configs/default.yaml \
  --hgt-artifacts outputs/hgt \
  --enable-generator \
  --host 127.0.0.1 --port 8000
```

不需要 HGT 时删除 `--hgt-artifacts`；只检索证据、不生成答案时删除 `--enable-generator`。

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the main contribution?"}'
```

请求体必须是 JSON。响应中的 `answer` 在未启用生成时为 `null`，`forest` 和 `total_cost` 仍可用于检索评测。

默认 INFO 日志显示模型加载、候选召回、HGT、Reranker、森林检索和生成阶段。公开数据评测按系统记录开始与完成，并每 50 个问题报告一次进度；无需额外参数。

## 7. 最小验收

```bash
paper-rag validate-config configs/default.yaml
paper-rag inspect-graph data/parsed/evidence_graph.json
curl http://127.0.0.1:8000/health
```

还应确认：

- Qdrant 与 NPZ 的基础向量维度为 2048；
- HGT `training.json` 的图哈希与当前图一致；
- `relation_triples` 大于 0 只表示关系损失有训练样本；若要声称多模态关系监督，还应确认图中存在 `caption_of`、`refers_to` 或 `derived_from`；
- 返回的 `total_cost` 不超过配置预算；
- GPU 模型、MinerU 输出质量和外部 API 需在实际 Linux 机器上验证。
