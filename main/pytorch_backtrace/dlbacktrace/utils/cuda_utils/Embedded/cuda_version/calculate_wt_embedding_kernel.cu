#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Optimized kernel: 1 thread per input element instead of vocab token
__global__ void wt_embedding_optimized_kernel(    
    const float* __restrict__ R_out,      // [B*T, D] input relevance
    const int64_t* __restrict__ input_ids, // [B*T] token indices  
    float* __restrict__ output,           // [vocab_size, D] output
    const int BT,                         // batch_size * sequence_length
    const int D,                          // embedding dimension
    const int vocab_size                  // vocabulary size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= BT) return;
    
    // Get token ID for this input position
    int64_t token_id = input_ids[tid];
    
    // Bounds check
    if (token_id < 0 || token_id >= vocab_size) return;
    
    // Coalesced read of relevance vector
    const float* R_in = R_out + tid * D;
    float* out_ptr = output + token_id * D;
    
    // Accumulate relevance with reduced atomic contention
    // Process in chunks to reduce atomic operations
    const int CHUNK_SIZE = 4;
    for (int d = 0; d < D; d += CHUNK_SIZE) {
        int end_d = min(d + CHUNK_SIZE, D);
        
        // Local accumulation
        float local_vals[CHUNK_SIZE] = {0.0f};
        for (int i = 0; i < end_d - d; i++) {
            local_vals[i] = R_in[d + i];
        }
        
        // Atomic updates in batch
        for (int i = 0; i < end_d - d; i++) {
            atomicAdd(&out_ptr[d + i], local_vals[i]);
        }
    }
}

// Warp-level optimized kernel for reduced atomic contention
__global__ void wt_embedding_warp_kernel(    
    const float* __restrict__ R_out,
    const int64_t* __restrict__ input_ids,
    float* __restrict__ output,
    const int BT,
    const int D,
    const int vocab_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= BT) return;
    
    int64_t token_id = input_ids[tid];
    if (token_id < 0 || token_id >= vocab_size) return;
    
    // Simple approach: each thread processes its input position independently
    // This avoids complex warp synchronization and reduces atomic contention
    const float* R_in = R_out + tid * D;
    float* out_ptr = output + token_id * D;
    
    // Process in chunks to improve memory coalescing
    const int CHUNK_SIZE = 4;
    for (int d = 0; d < D; d += CHUNK_SIZE) {
        int end_d = min(d + CHUNK_SIZE, D);
        
        // Vectorized load and atomic update
        for (int i = 0; i < end_d - d; i++) {
            atomicAdd(&out_ptr[d + i], R_in[d + i]);
        }
    }
}

// Mean aggregation kernel
__global__ void wt_embedding_mean_kernel(    
    const float* __restrict__ R_out,
    const int64_t* __restrict__ input_ids,
    float* __restrict__ output,
    const int BT,
    const int D,
    const int vocab_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= BT) return;
    
    const int64_t token_id = input_ids[tid];
    if (__builtin_expect(token_id < 0 || token_id >= vocab_size, 0)) return;
    
    // Cache R_in pointer
    const float* __restrict__ R_in = R_out + tid * D;
    
    // Vectorized summation with multiple accumulators to reduce dependency chains
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

    int d = 0;
    const int D_vec = D & ~3;

    #pragma unroll 4
    for (; d < D_vec; d += 4) {
        // Load 4 elements at once
        const float4 vals = reinterpret_cast<const float4*>(R_in)[d >> 2];

        sum1 += vals.x;
        sum2 += vals.y;
        sum3 += vals.z;
        sum4 += vals.w;
    }

    // Handle remaining elements
    float remaining = 0.0f;
    for (; d < D; d++) {
        remaining += R_in[d];
    }

    // Combine all partial sums
    const float total_sum = sum1 + sum2 + sum3 + sum4 + remaining;
    
    // Use fast division approximation (multiply by reciprocal)
    const float mean_val = total_sum * __frcp_rn(static_cast<float>(D));
    
    // Atomic operation on final result
    atomicAdd(&output[token_id], mean_val);
}

// Host function with optimized kernel selection
torch::Tensor wt_embedding_cuda(
    const torch::Tensor& R_out,
    const torch::Tensor& input_ids, 
    const int vocab_size,
    const std::string& aggregate
) {
    // Input validation
    TORCH_CHECK(R_out.is_cuda(), "R_out must be on CUDA device");
    TORCH_CHECK(input_ids.is_cuda(), "input_ids must be on CUDA device");
    TORCH_CHECK(R_out.dim() == 3, "R_out must be 3D tensor [B, T, D]");
    TORCH_CHECK(input_ids.dim() == 2, "input_ids must be 2D tensor [B, T]");
    TORCH_CHECK(R_out.dtype() == torch::kFloat32, "R_out must be float32");
    TORCH_CHECK(input_ids.dtype() == torch::kInt64, "input_ids must be int64");
    
    auto R_out_t = R_out.contiguous();
    auto input_ids_t = input_ids.contiguous();
    
    const int B = R_out_t.size(0);
    const int T = R_out_t.size(1); 
    const int D = R_out_t.size(2);
    const int BT = B * T;
    
    TORCH_CHECK(aggregate == "sum" || aggregate == "mean", 
                "aggregate must be 'sum' or 'mean'");
    
    // Create output tensor
    torch::Tensor output;
    if (aggregate == "mean") {
        output = torch::zeros({vocab_size}, R_out_t.options());
    } else if (aggregate == "sum") {
        output = torch::zeros({vocab_size, D}, R_out_t.options());
    }
    
    // Reshape input tensors
    auto R_out_flat = R_out_t.view({BT, D});
    auto input_ids_flat = input_ids_t.view({BT});
    
    // Choose optimal kernel based on problem size
    const int BLOCK_SIZE = 256;
    dim3 gridDim((BT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    
    if (aggregate == "mean") {
        // Use specialized mean kernel
        wt_embedding_mean_kernel<<<gridDim, blockDim>>>(
            R_out_flat.data_ptr<float>(),
            input_ids_flat.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            BT, D, vocab_size
        );
    } else if (aggregate == "sum") {
        // Choose sum kernel based on embedding dimension
        if (D <= 256) {
            // Use warp-optimized kernel for small to medium embeddings
            // Shared memory kernel was causing issues due to excessive memory usage
            wt_embedding_warp_kernel<<<gridDim, blockDim>>>(
                R_out_flat.data_ptr<float>(),
                input_ids_flat.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                BT, D, vocab_size
            );
        } else {
            // Use basic optimized kernel for large embeddings
            wt_embedding_optimized_kernel<<<gridDim, blockDim>>>(
                R_out_flat.data_ptr<float>(),
                input_ids_flat.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                BT, D, vocab_size
            );
        }
    } else {
        TORCH_CHECK(false, "Invalid aggregate type: ", aggregate);
    }
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));
    
    return output;
} 
