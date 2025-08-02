import torch

# Linear Layer
from .cuda_utils.Linear_v2.original_version import calculate_wt_fc as calculate_wt_fc_original_linear
from .cuda_utils.Linear_v2.pytorch_version import calculate_wt_fc as calculate_wt_fc_pytorch_linear
from .cuda_utils.Linear_v2.cuda_v2 import calculate_wt_fc_cuda as calculate_wt_fc_cuda_linear

# Conv2D Layer - Under Development
from .cuda_utils.Conv2D.original_version import calculate_wt_conv as calculate_wt_conv_original
from .cuda_utils.Conv2D.refactored_version import calculate_wt_conv as calculate_wt_conv_refactored
from .cuda_utils.Conv2D.pytorch_version import calculate_wt_conv as calculate_wt_conv_pytorch
from .cuda_utils.Conv2D.cuda_v2 import calculate_wt_conv_cuda as calculate_wt_conv_cuda

# MaxPool2D Layer
from .cuda_utils.MaxPool2D.original_version import calculate_wt_maxpool as calculate_wt_maxpool_original
from .cuda_utils.MaxPool2D.refactored_version import calculate_wt_maxpool as calculate_wt_maxpool_refactored
from .cuda_utils.MaxPool2D.pytorch_version import calculate_wt_maxpool as calculate_wt_maxpool_pytorch
#from .cuda_utils.MaxPool2D.cuda_version import calculate_wt_maxpool_cuda as calculate_wt_maxpool_cuda

# AdaptiveAvgPool2D Layer
from .cuda_utils.AdaptiveAvgPool2D.original_version import calculate_wt_gavgpool as calculate_wt_gavgpool_original
from .cuda_utils.AdaptiveAvgPool2D.refactored_version import calculate_wt_gavgpool as calculate_wt_gavgpool_refactored
from .cuda_utils.AdaptiveAvgPool2D.pytorch_version import calculate_wt_gavgpool as calculate_wt_gavgpool_pytorch
#from .cuda_utils.AdaptiveAvgPool2D.cuda_version.wt_gavgpool_ops import fused_weighted_gavgpool as calculate_wt_gavgpool_cuda

# Embedded Layer
from .cuda_utils.Embedded.original_version import calculate_wt_embedding as calculate_wt_embedding_original
from .cuda_utils.Embedded.refactored_version import calculate_wt_embedding as calculate_wt_embedding_refactored
from .cuda_utils.Embedded.pytorch_version import calculate_wt_embedding as calculate_wt_embedding_pytorch
#from .cuda_utils.Embedded.cuda_version.wt_embedding_ops import wt_embedding_cuda as calculate_wt_embedding_cuda

# SelfAttention Layer - Under Development
from .cuda_utils.SelfAttention.original_version import calculate_wt_self_attention as calculate_wt_self_attention_original
from .cuda_utils.SelfAttention.pytorch_version import calculate_wt_self_attention as calculate_wt_self_attention_pytorch
#from .cuda_utils.SelfAttention.cuda_version.wt_selfattention_ops import wt_selfattention_cuda as calculate_wt_self_attention_cuda

# Wt_add_equal Layer
from .cuda_utils.Wt_add_equal.original_version import calculate_wt_add_equal as calculate_wt_add_original
from .cuda_utils.Wt_add_equal.refactored_version import calculate_wt_add_equal as calculate_wt_add_refactored
from .cuda_utils.Wt_add_equal.pytorch_version import calculate_wt_add_equal_vectorized as calculate_wt_add_pytorch
#from .cuda_utils.Wt_add_equal.cuda_version.wt_add_equal_ops import wt_add_equal_cuda as calculate_wt_add_cuda

# Wt_mul Layer
from .cuda_utils.Wt_mul.original_version import calculate_wt_mul as calculate_wt_mul_original
from .cuda_utils.Wt_mul.refactored_version import calculate_wt_mul as calculate_wt_mul_refactored

def _prepare_tensors(device, *arrays):
    return [torch.tensor(arr, dtype=torch.float32, device=device) for arr in arrays]

def launch_linear(version, wts, inp, w, b, act):
    if version == 'original':
        func = calculate_wt_fc_original_linear
        return func(wts, inp, w, b, act)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if version == 'pytorch':
        wts_t, inp_t, w_t, b_t = _prepare_tensors(device, wts, inp, w, b)
        return calculate_wt_fc_pytorch_linear(wts_t, inp_t, w_t, b_t, act)
    elif version == 'cuda':
        try:
            # Try CUDA implementation with error handling
            result = calculate_wt_fc_cuda_linear(wts, inp, w, b, act)
            if result is None:
                print(f"⚠️  CUDA linear implementation returned None, falling back to original")
                return calculate_wt_fc_original_linear(wts, inp, w, b, act)
            return result
        except Exception as e:
            print(f"⚠️  CUDA linear implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return calculate_wt_fc_original_linear(wts, inp, w, b, act)
    else:
        raise ValueError(f"Unknown version for Linear layer: {version}")

def launch_conv2d(version, wts, inp, w, b, padding, strides, act):
    if version in ['original', 'refactored']:
        func = calculate_wt_conv_original if version == 'original' else calculate_wt_conv_refactored
        return func(wts, inp, w, b, padding, strides, act)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if version == 'pytorch':
        return calculate_wt_conv_pytorch(wts, inp, w, b, padding, strides, act)
    
    elif version == 'cuda':
        return calculate_wt_conv_cuda(wts, inp, w, b, padding, strides, act).cpu().numpy()

    else:
        raise ValueError(f"Unknown version for Conv2D layer: {version}")

def launch_maxpool2d(version, wts, inp, pool_size, padding, strides):
    if version in ['original', 'refactored']:
        func = calculate_wt_maxpool_original if version == 'original' else calculate_wt_maxpool_refactored
        return func(wts, inp, pool_size, padding, strides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wts_t, inp_t = _prepare_tensors(device, wts, inp)

    if version == 'pytorch':
        return calculate_wt_maxpool_pytorch(wts_t, inp_t, pool_size, padding, strides).cpu().numpy()
    elif version == 'cuda':
        #return calculate_wt_maxpool_cuda(wts_t, inp_t, pool_size, padding, strides).cpu().numpy()
        return None
    else:
        raise ValueError(f"Unknown version for MaxPool2D layer: {version}")

def launch_adaptiveavgpool2d(version, wts, inp):
    if version in ['original', 'refactored']:
        func = calculate_wt_gavgpool_original if version == 'original' else calculate_wt_gavgpool_refactored
        return func(wts, inp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wts_t, inp_t = _prepare_tensors(device, wts, inp)

    if version == 'pytorch':
        return calculate_wt_gavgpool_pytorch(wts_t, inp_t)[0].cpu().numpy()
    elif version == 'cuda':
        #return calculate_wt_gavgpool_cuda(wts_t, inp_t)[0].cpu().numpy()
        return None
    else:
        raise ValueError(f"Unknown version for AdaptiveAvgPool2D layer: {version}")

def launch_embedding(version, R_out, inp, vocab_size, aggregate):
    if version in ['original', 'refactored']:
        func = calculate_wt_embedding_original if version == 'original' else calculate_wt_embedding_refactored
        return func(R_out, inp, vocab_size, aggregate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_out_t = torch.tensor(R_out, dtype=torch.float32, device=device)
    inp_t = torch.tensor(inp, dtype=torch.long, device=device)

    if version == 'pytorch':
        return calculate_wt_embedding_pytorch(R_out_t, inp_t, vocab_size, aggregate)[0].cpu().numpy()
    elif version == 'cuda':
        #return calculate_wt_embedding_cuda(R_out_t, inp_t, vocab_size, aggregate)[0].cpu().numpy()
        return None
    else:
        raise ValueError(f"Unknown version for Embedding layer: {version}")

def launch_self_attention(version, R_out, Q, K, V, masked_fill=None, scale=None, epsilon=1e-9):
    if version == 'original':
        return calculate_wt_self_attention_original(R_out, Q, K, V, masked_fill, scale, epsilon)

    masked_fill_t = torch.tensor(masked_fill, dtype=torch.float32, device=device) if masked_fill is not None else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_out_t, Q_t, K_t, V_t, scale_t = _prepare_tensors(device, R_out, Q, K, V, scale, epsilon)
    
    if version == 'pytorch':
        result_torch = calculate_wt_self_attention_pytorch(R_out_t, Q_t, K_t, V_t, masked_fill_t, scale_t, epsilon)
        return [arr.cpu().numpy() for arr in result_torch]
    elif version == 'cuda':
        #result_cuda = calculate_wt_self_attention_cuda(R_out_t, Q_t, K_t, V_t, masked_fill_t, scale_t, epsilon)
        #return [arr.cpu().numpy() for arr in result_cuda]
        return None
    else:
        raise ValueError(f"Unknown version for SelfAttention layer: {version}")

def launch_wt_add_equal(version, R_out, inp):
    if version in ['original', 'refactored']:
        func = calculate_wt_add_original if version == 'original' else calculate_wt_add_refactored
        return func(R_out, inp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_out_t, inp_t = _prepare_tensors(device, R_out, inp)

    if version == 'pytorch':
        return [arr.cpu().numpy() for arr in calculate_wt_add_pytorch(R_out_t, inp_t)[0]]
    elif version == 'cuda':
        #return [arr.cpu().numpy() for arr in calculate_wt_add_cuda(R_out_t, inp_t)[0]]
        return None
    else:
        raise ValueError(f"Unknown version for Wt_add_equal layer: {version}")

def launch_wt_mul(version, R_out, X, Y):
    if version in ['original', 'refactored']:
        func = calculate_wt_mul_original if version == 'original' else calculate_wt_mul_refactored
        return func(R_out, X, Y)
    else:
        # Fallback to original for unsupported implementations
        print(f"⚠️  {version} implementation not available for Wt_mul, using original")
        return calculate_wt_mul_original(R_out, X, Y)
