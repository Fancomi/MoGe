#!/bin/bash
# MoGe 一键安装脚本
# 用法: bash install.sh [环境路径] [代理选择: baidu|aliyun]
set -e

ENV_DIR="${1:-/root/paddlejob/workspace/env_run/penghaotian/envs/moge}"
PROXY="${2:-baidu}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 代理配置
if [ "$PROXY" = "aliyun" ]; then
    export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    export http_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
else
    export https_proxy=http://agent.baidu.com:8188
    export http_proxy=http://agent.baidu.com:8188
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
fi

echo "[1/4] 创建虚拟环境: $ENV_DIR"
uv venv "$ENV_DIR" --python 3.10 2>/dev/null || true

PYTHON="$ENV_DIR/bin/python"
UV_INSTALL="uv pip install --python $PYTHON --link-mode=copy"

echo "[2/4] 安装 PyTorch (CUDA 12.4)"
$UV_INSTALL torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[3/4] 安装依赖"
$UV_INSTALL click opencv-python scipy matplotlib trimesh pillow huggingface_hub numpy -i "$PIP_INDEX"
$UV_INSTALL ultralytics timm -i "$PIP_INDEX"
$UV_INSTALL "git+https://github.com/ultralytics/CLIP.git"
$UV_INSTALL "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
$UV_INSTALL "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"

echo "[4/4] 安装 MoGe (editable)"
$UV_INSTALL -e "$SCRIPT_DIR" --no-deps

echo "安装完成! 激活环境: source $ENV_DIR/bin/activate"
