#!/usr/bin/env bash
set -euo pipefail

mkdir -p minitorch/cuda_kernels

# Override with CUDA_ARCH_FLAGS if needed, e.g.:
# CUDA_ARCH_FLAGS="-gencode arch=compute_70,code=sm_70" bash compile_cuda.sh
CUDA_ARCH_FLAGS=${CUDA_ARCH_FLAGS:--arch=sm_86}

nvcc $CUDA_ARCH_FLAGS -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc $CUDA_ARCH_FLAGS -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
nvcc $CUDA_ARCH_FLAGS -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
nvcc $CUDA_ARCH_FLAGS -o minitorch/cuda_kernels/flash_attention_kernel.so --shared src/flash_attention_kernel.cu -Xcompiler -fPIC
