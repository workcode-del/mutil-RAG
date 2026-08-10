#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "未检测到已激活的Conda环境，请先执行 conda activate <环境名>。" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "请先激活项目专用Conda环境，不要把依赖安装到base环境。" >&2
  exit 1
fi

active_environment="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

bash scripts/configure_tuna_sources.sh

if ! python -c "import torch, torchvision; assert torch.__version__.startswith('2.8.'); assert torchvision.__version__.startswith('0.23.')" 2>/dev/null; then
  echo "未检测到匹配版本，开始从当前清华PyPI源安装PyTorch固定版本。"
  python -m pip install \
    --prefer-binary \
    --retries 5 \
    --timeout 120 \
    -r requirements/torch.txt
fi

python -c "import torch, torchvision; assert torch.__version__.startswith('2.8.'), torch.__version__; assert torchvision.__version__.startswith('0.23.'), torchvision.__version__"

python -m pip install \
  --prefer-binary \
  --disable-pip-version-check \
  --retries 5 \
  --timeout 120 \
  -r requirements/locked.txt

python -c "import transformers; assert transformers.__version__ == '4.57.6', transformers.__version__"

python -m pip install --no-deps -e .

qwen_repo="$project_root/third_party/Qwen3-VL-Embedding"
if [[ -d "$qwen_repo" ]]; then
  python -m pip install --no-deps -e "$qwen_repo"
else
  echo "警告：尚未找到third_party/Qwen3-VL-Embedding。" >&2
  echo "克隆后执行：python -m pip install --no-deps -e third_party/Qwen3-VL-Embedding" >&2
fi

python -m pip check
echo "Conda环境 '${active_environment}' 的固定依赖安装完成。"
