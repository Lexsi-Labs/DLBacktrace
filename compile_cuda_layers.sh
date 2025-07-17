#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting compilation of all CUDA layers..."

# Base directory for layers
LAYERS_DIR="dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/layers"

# Check if the layers directory exists
if [ ! -d "$LAYERS_DIR" ]; then
    echo "Error: Layers directory not found at '$LAYERS_DIR'"
    exit 1
fi

# Find all setup.py files within the layers directory and iterate
find "$LAYERS_DIR" -name "setup.py" -print0 | while IFS= read -r -d $'\0' setup_file; do
    setup_dir=$(dirname "$setup_file")
    echo "==> Compiling in $setup_dir"

    # Execute compilation in a subshell to isolate directory changes
    (
        echo "Changing directory to $setup_dir"
        cd "$setup_dir"
        echo "Running 'python setup.py develop'..."
        /home/omkar/Brendan/Projects/Backtrace_layers/DL-Backtrace/myvenv/bin/python setup.py develop
    )

    # Check if the subshell command was successful
    # The exit code of the subshell is the exit code of the last command executed within it.
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed in $setup_dir"
        exit 1
    fi

    echo "<== Finished compiling in $setup_dir"
    echo "----------------------------------------"
done

echo "All CUDA layers compiled successfully!" 
