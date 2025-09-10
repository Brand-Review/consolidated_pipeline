#!/bin/bash

# Safe Environment Setup Script
# Prevents semaphore leaks and multiprocessing issues

echo "🔧 Setting up safe environment for BrandGuard Pipeline..."

# Export environment variables to prevent multiprocessing issues
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

# Disable Python multiprocessing warnings
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing"

echo "✅ Environment variables set:"
echo "   TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "   NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   PYTHONWARNINGS=$PYTHONWARNINGS"

echo ""
echo "🚀 You can now run the server safely:"
echo "   python start_server_safe.py"
echo "   or"
echo "   python app.py"
echo ""
