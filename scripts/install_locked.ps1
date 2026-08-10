$ErrorActionPreference = "Stop"

if (-not $env:CONDA_PREFIX) {
    throw "未检测到已激活的Conda环境，请先执行 conda activate <环境名>。"
}

$activeEnvironment = if ($env:CONDA_DEFAULT_ENV) { $env:CONDA_DEFAULT_ENV } else { Split-Path -Leaf $env:CONDA_PREFIX }
if ($activeEnvironment -eq "base") {
    throw "请先激活项目专用Conda环境，不要把依赖安装到base环境。"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

powershell -ExecutionPolicy Bypass -File scripts/configure_tuna_sources.ps1

python -c "import torch, torchvision; assert torch.__version__.startswith('2.8.'); assert torchvision.__version__.startswith('0.23.')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "未检测到匹配版本，开始从当前清华PyPI源安装PyTorch固定版本。"
    python -m pip install `
        --prefer-binary `
        --retries 5 `
        --timeout 120 `
        -r requirements/torch.txt
}

python -c "import torch, torchvision; assert torch.__version__.startswith('2.8.'), torch.__version__; assert torchvision.__version__.startswith('0.23.'), torchvision.__version__"

python -m pip install `
    --prefer-binary `
    --disable-pip-version-check `
    --retries 5 `
    --timeout 120 `
    -r requirements/locked.txt

python -c "import transformers; assert transformers.__version__ == '4.57.6', transformers.__version__"

# 项目自身及Qwen官方适配仓库不再触发第二轮依赖求解。
python -m pip install --no-deps -e .

$qwenRepo = Join-Path $projectRoot "third_party/Qwen3-VL-Embedding"
if (Test-Path $qwenRepo) {
    python -m pip install --no-deps -e $qwenRepo
} else {
    Write-Warning "尚未找到third_party/Qwen3-VL-Embedding；克隆后执行："
    Write-Warning "python -m pip install --no-deps -e third_party/Qwen3-VL-Embedding"
}

python -m pip check
Write-Host "Conda环境 '$activeEnvironment' 的固定依赖安装完成。"
