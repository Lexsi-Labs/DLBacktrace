#!/bin/bash
ulimit -v 24000000  
# Generate timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Define folder and file paths
log_dir="densenet_logs/$timestamp"
log_file="$log_dir/trace_log_$timestamp.log"
err_file="$log_dir/trace_error_$timestamp.log"

# Create the log directory
mkdir -p "$log_dir"

# Run the Python script with output and error logs
python dlb_densenet.py > "$log_file" 2> "$err_file"

# Status message
echo "✅ Run completed."
echo "📄 Logs saved to → $log_file"
echo "⚠️  Errors saved to → $err_file"

