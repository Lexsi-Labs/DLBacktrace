import time
import numpy as np
from original_version import calculate_wt_fc as calculate_wt_fc_original
from refactored_version import calculate_wt_fc as calculate_wt_fc_refactored
from pytroch_version import calculate_wt_fc as calculate_wt_fc_pytorch, ActivationParams, ActivationRange
import torch

def generate_random_data(input_size, output_size):
    """Generates random data for benchmarking."""
    inp = np.random.normal(loc=0.0, scale=1.0, size=(input_size,)).astype(np.float32)
    w = np.random.normal(loc=0.0, scale=0.05, size=(output_size, input_size)).astype(np.float32)
    b = np.random.normal(loc=0.0, scale=0.01, size=(output_size,)).astype(np.float32)
    wts = np.random.normal(loc=0.0, scale=1.0, size=(output_size,)).astype(np.float32)
    # Example activation function dictionary, adjust as needed
    act = {"type": "mono", "range": {"l": -4, "u": 4}, "func": None} 
    return wts, inp, w, b, act

def run_benchmark(input_size, output_size, num_runs=100):
    """Runs the benchmark and compares the two functions."""
    wts, inp, w, b, act = generate_random_data(input_size, output_size)

    # Benchmark original version
    start_time_orig = time.time()
    for _ in range(num_runs):
        output_orig = calculate_wt_fc_original(wts, inp, w, b, act)
    print(output_orig.dtype)
    end_time_orig = time.time()
    time_orig = (end_time_orig - start_time_orig) / num_runs
    
    print(f"Original function average time: {time_orig:.6f} seconds")
    
        # Benchmark refactored version
    start_time_refactored = time.time()
    for _ in range(num_runs):
        output_refactored = calculate_wt_fc_refactored(wts, inp, w, b, act)
    print(output_refactored.dtype)
    end_time_refactored = time.time()
    time_refactored = (end_time_refactored - start_time_refactored) / num_runs

    print(f"Refactored function average time: {time_refactored:.6f} seconds")

    wts_torch = torch.tensor(wts, dtype=torch.float32)
    inp_torch = torch.tensor(inp, dtype=torch.float32)
    w_torch = torch.tensor(w, dtype=torch.float32)
    b_torch = torch.tensor(b, dtype=torch.float32)
    act_torch = ActivationParams(type="mono", range=ActivationRange(l=-4.0, u=4.0), func=None)
    
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wts_torch = wts_torch.to(device=cuda_device)
    inp_torch = inp_torch.to(device=cuda_device)
    w_torch = w_torch.to(device=cuda_device)
    b_torch = b_torch.to(device=cuda_device)
    
    # Benchmark PyTorch version
    start_time_pytorch = time.time()
    for _ in range(num_runs):
        output_pytorch = calculate_wt_fc_pytorch(wts_torch, inp_torch, w_torch, b_torch, act_torch)
    print(output_pytorch.dtype)
    end_time_pytorch = time.time()
    time_pytorch = (end_time_pytorch - start_time_pytorch) / num_runs
    
    output_pytorch = output_pytorch.cpu().numpy()
    
    print(f"PyTorch function average time: {time_pytorch:.6f} seconds")
    
    # Compare outputs
    outputs_match = np.allclose(output_orig, output_refactored,rtol=1e-03, atol=1e-05)
    outputs_match_pytorch = np.allclose(output_orig, output_pytorch,rtol=1e-03, atol=1e-05)

    print(f"--- Benchmark Results (input_size={input_size}, output_size={output_size}, num_runs={num_runs}) ---")
    
    if time_refactored > 0:
        speedup = time_orig / time_refactored
        print(f"Speedup (Refactored vs Original): {speedup:.2f}x")
    else:
        print("Speedup: N/A (Refactored version was too fast to measure or did not run)")

    if time_pytorch > 0:
        speedup_pytorch = time_orig / time_pytorch
        print(f"Speedup (PyTorch vs Original): {speedup_pytorch:.2f}x")
    else:
        print("Speedup: N/A (PyTorch version was too fast to measure or did not run)")
    
    print(output_orig)
    print(output_refactored)
    print(output_pytorch)

    print(f"Outputs match: {outputs_match}")
    print(f"Outputs match PyTorch: {outputs_match_pytorch}")
    
    if not outputs_match:
        print("Outputs differ:")
        print(f"  Original output: {output_orig}")
        print(f"  Refactored output: {output_refactored}")
        # Optionally, print the difference
        print(f"  Difference: {output_orig - output_refactored}")

    return time_orig, time_refactored, time_pytorch, outputs_match, outputs_match_pytorch

if __name__ == "__main__":
    # Define input and output sizes for the test data
    # These are example values, please adjust them to realistic sizes for your use case.
    test_input_size = 4096
    test_output_size = 1024
    num_benchmark_runs = 100

    print("Running benchmark...")
    run_benchmark(test_input_size, test_output_size, num_benchmark_runs)
    print("\nBenchmark complete.")


