#!/bin/bash

# 1. Load correct CUDA module (must match Megatron worker venv: PyTorch uses cu12)
# conda deactivate
module purge
module load cudatoolkit/12.9

# 2. Set CUDA Home and Path
mkdir -p $HOME/libcudart_block
# create a deliberately-invalid shared library file
# printf "not a real so\n" > $HOME/libcudart_block/libcudart.so.13

# export LD_LIBRARY_PATH="$HOME/libcudart_block:$LD_LIBRARY_PATH"

# also ensure CUDA 12 is early in the path
export CUDA_HOME=/usr/local/cuda-12.9   # or your chosen 12.x
# export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
# unset CUDA_HOME CUDA_PATH
# export CUDA_HOME=$CUDA_ROOT 2>/dev/null || true
# export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
# CRITICAL: Do NOT append $LD_LIBRARY_PATH here - it can contain libcudart.so.13 from conda/modules,
# causing "Multiple libcudart libraries found" in Transformer Engine. We set it explicitly in step 5.

# 3. Detect Python paths for cuDNN and NCCL in the main venv
# Note: We use the main .venv to find these paths
PYTHON_EXEC=".venv/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: .venv/bin/python not found. Make sure you are in the project root."
    return 1
fi

export CUDNN_PATH=$($PYTHON_EXEC -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")
export NCCL_PATH=$($PYTHON_EXEC -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")

# 4. Set Compiler Flags (CPATH)
# This allows the compiler to find cudnn.h and nccl.h
export CPATH=$CUDNN_PATH/include:$NCCL_PATH/include:$CPATH

# 5. Set Linker Flags (LD_LIBRARY_PATH)
# Use ONLY our explicit paths - do NOT append inherited LD_LIBRARY_PATH (may contain libcudart.so.13 from conda/modules)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_PATH/lib:$NCCL_PATH/lib
# export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' \
#   | grep -vE '/usr/local/cuda(/|$)|/usr/local/cuda-13' \
#   | paste -sd: -)"

# 6. Set Transformer Engine Framework
export NVTE_FRAMEWORK=pytorch

# 7. Set Architecture (Optional, helps with build speed/errors)
export TORCH_CUDA_ARCH_LIST="9.0"

echo "Environment configured for NeMo-RL on Della."
echo "CUDNN_PATH: $CUDNN_PATH"
echo "NCCL_PATH: $NCCL_PATH"
