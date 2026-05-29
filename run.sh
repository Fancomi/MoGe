#!/bin/bash
# MoGe 预测脚本 — 用法: bash run.sh <输入路径> [输出目录] [选项...]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/root/paddlejob/workspace/env_run/penghaotian/envs/moge/bin/python"
INPUT="${1:?用法: bash run.sh <输入图片/视频> [输出目录] [其他参数...]}"
OUTPUT="${2:-./output}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

exec $PYTHON "$SCRIPT_DIR/predict.py" -i "$INPUT" -o "$OUTPUT" "$@"
