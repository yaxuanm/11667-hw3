#!/bin/bash

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 检查GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 运行训练
echo "Starting training..."
python src/calculator/main.py

echo "Training completed!"
