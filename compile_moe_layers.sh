#!/bin/bash
# compile_moe_layers.sh
# Build all precompiled CUDA extensions for the MoE PyTorch backend.
# Usage: bash compile_moe_layers.sh
#        (run from the project root, i.e. the directory containing this script)

set -e

echo "Starting compilation of all MoE CUDA kernels..."

LAYERS_DIR="dl_backtrace/moe_pytorch_backtrace/backtrace/utils/cuda_utils/MoE_utils/cuda_version"

if [ ! -d "$LAYERS_DIR" ]; then
    echo "Error: MoE cuda_version directory not found at '$LAYERS_DIR'"
    exit 1
fi

find "$LAYERS_DIR" -name "setup.py" -print0 | while IFS= read -r -d $'\0' setup_file; do
    setup_dir=$(dirname "$setup_file")
    echo "==> Compiling in $setup_dir"

    (
        cd "$setup_dir"
        echo "Running 'python3 setup.py develop'..."
        python3 setup.py develop
    )

    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed in $setup_dir"
        exit 1
    fi

    echo "<== Finished compiling in $setup_dir"
    echo "----------------------------------------"
done

echo "All MoE CUDA kernels compiled successfully!"
