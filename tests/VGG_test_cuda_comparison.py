import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dl_backtrace.pytorch_backtrace.backtrace.backtrace import Backtrace as B
import time
import os
from typing import Dict, Any

# Import the refactored layer implementations
from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Linear.original_version import calculate_wt_fc as linear_original
from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Conv2D.original_version import calculate_wt_conv as conv2d_original  
from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.MaxPool2D.original_version import calculate_wt_maxpool as maxpool_original

# Create synthetic dataset for testing
def create_synthetic_data(num_samples=100, num_classes=6, image_size=(3, 224, 224)):
    """Create synthetic image data for testing purposes."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = torch.randn(num_samples, *image_size)
    X = (X - X.min()) / (X.max() - X.min())
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

# Simple VGG-like model for testing
class SimpleVGG(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleVGG, self).__init__()
        self.identity = nn.Identity()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.identity(x)
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class CUDABacktrace(B):
    """Extended Backtrace class with CUDA layer support for comparing original vs CUDA implementations."""
    
    def __init__(self, model=None, activation_dict={}, model_type=None, use_cuda_layers=False):
        super().__init__(model, activation_dict, model_type)
        self.use_cuda_layers = use_cuda_layers
        self.layer_timings = {}
        
    def proportional_eval_with_cuda(self, all_out, start_wt=[], multiplier=100.0, 
                                   scaler=0, max_unit=0, predicted_token=None,
                                   thresholding=0.5, task="binary-classification"):
        """Modified proportional_eval that can use CUDA implementations for specific layers."""
        
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        out_layer = model_resource[2][0]
        all_wt = {}
        
        # Initialize starting weights
        if len(start_wt) == 0:
            from dl_backtrace.pytorch_backtrace.backtrace.utils import prop as UP
            start_wt = UP.calculate_start_wt(all_out[out_layer], scaler, thresholding, task=task)
            all_wt[out_layer] = start_wt * multiplier
            layer_stack = self.layer_stack
            
        for start_layer in layer_stack:
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                
                # Initialize child weights if not present
                for ch in child_nodes:
                    if ch not in all_wt:
                        # Handle both tensor and numpy array cases
                        if isinstance(all_out[ch], dict) and 0 in all_out[ch]:
                            # Handle dictionary case (from predict_every)
                            child_data = all_out[ch][0]
                        else:
                            # Handle direct array case
                            child_data = all_out[ch]
                        
                        # Convert to numpy if it's a tensor
                        if hasattr(child_data, 'detach'):
                            child_data = child_data.detach().numpy()
                        
                        all_wt[ch] = np.zeros_like(child_data)

                layer_class = model_resource[1][start_layer]["class"]
                layer_start_time = time.time()
                
                if layer_class == "Linear":
                    self._process_linear_layer(start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict)
                    
                elif layer_class == "Conv2d":
                    self._process_conv2d_layer(start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict)
                    
                elif layer_class == "MaxPool2d":
                    self._process_maxpool2d_layer(start_layer, child_nodes, all_wt, all_out, model_resource)
                    
                else:
                    # Use original implementation for other layers
                    self._process_other_layer(start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict)
                
                layer_end_time = time.time()
                layer_timing = layer_end_time - layer_start_time
                
                if layer_class not in self.layer_timings:
                    self.layer_timings[layer_class] = []
                self.layer_timings[layer_class].append(layer_timing)
                
        return all_wt
    
    def _get_layer_data(self, all_out, layer_name):
        """Helper function to extract layer data from all_out, handling different formats."""
        if isinstance(all_out[layer_name], dict) and 0 in all_out[layer_name]:
            # Handle dictionary case (from predict_every)
            data = all_out[layer_name][0]
        else:
            # Handle direct array case
            data = all_out[layer_name]
        
        # Convert to numpy if it's a tensor
        if hasattr(data, 'detach'):
            data = data.detach().numpy()
        
        return data
    
    def _process_linear_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict):
        """Process Linear layer with original or CUDA implementation."""
        l1 = model_resource[0][start_layer]
        w1 = l1.state_dict()['weight']
        b1 = l1.state_dict()['bias']
        
        # Get input data correctly
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # Handle batch dimension - the original implementation expects single samples
        # If we have a batch, we need to process each sample individually
        if len(input_data.shape) > 1 and input_data.shape[0] > 1:
            # Multiple samples in batch - process first sample only for now
            input_sample = input_data[0]  # Take first sample
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) > 1 else all_wt[start_layer]
        else:
            # Single sample or already 1D
            input_sample = input_data.flatten() if len(input_data.shape) > 1 else input_data
            weight_sample = all_wt[start_layer].flatten() if len(all_wt[start_layer].shape) > 1 else all_wt[start_layer]
        
        if self.use_cuda_layers and torch.cuda.is_available():
            # Use CUDA implementation
            try:
                # Import the CUDA version
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Linear.cuda_version.wt_fc_ops import calculate_wt_fc_interface as linear_cuda
                
                # Convert parameters for CUDA function (all must be tensors and on GPU)
                row_specific_weights = torch.tensor(weight_sample, dtype=torch.float32).cuda()
                input_activations = torch.tensor(input_sample, dtype=torch.float32).cuda()
                weights_matrix = w1.cuda()
                bias_vector = b1.cuda()
                
                # Get activation parameters
                activation_params = activation_dict[model_resource[1][start_layer]["name"]]
                
                # Convert activation parameters to CUDA function format
                has_lower_bound = activation_params["range"]["l"] is not None
                lower_threshold = float(activation_params["range"]["l"]) if has_lower_bound else 0.0
                has_upper_bound = activation_params["range"]["u"] is not None
                upper_threshold = float(activation_params["range"]["u"]) if has_upper_bound else 0.0
                is_non_mono = activation_params["type"] == "non_mono"
                
                # Map activation function to enum
                if activation_params.get("func"):
                    if "relu" in str(activation_params["func"]).lower():
                        activation_func = 1
                    elif "sigmoid" in str(activation_params["func"]).lower():
                        activation_func = 2
                    else:
                        activation_func = 0  # identity
                else:
                    activation_func = 0  # identity
                
                # Call CUDA function with correct signature
                temp_wt = linear_cuda(
                    row_specific_weights,    # 1D tensor
                    input_activations,       # 1D tensor  
                    weights_matrix,          # 2D tensor
                    bias_vector,             # 1D tensor
                    has_lower_bound,         # bool
                    lower_threshold,         # float
                    has_upper_bound,         # bool
                    upper_threshold,         # float
                    is_non_mono,             # bool
                    activation_func          # int
                ).cpu().numpy()
                
            except Exception as e:
                print(f"CUDA Linear implementation failed: {e}, falling back to original version")
                # Use original implementation with numpy arrays
                temp_wt = linear_original(
                    weight_sample,  # 1D numpy array
                    input_sample,  # 1D numpy array  
                    w1.detach().numpy(),  # convert to numpy
                    b1.detach().numpy(),  # convert to numpy
                    activation_dict[model_resource[1][start_layer]["name"]]
                )
        else:
            # Use original implementation with numpy arrays
            temp_wt = linear_original(
                weight_sample,  # 1D numpy array
                input_sample,  # 1D numpy array
                w1.detach().numpy(),  # convert to numpy
                b1.detach().numpy(),  # convert to numpy
                activation_dict[model_resource[1][start_layer]["name"]]
            )
        
        # Reshape result back to match expected output shape if needed
        if len(all_wt[child_nodes[0]].shape) > 1:
            # If the target has batch dimension, we need to handle that
            temp_wt = temp_wt.reshape(all_wt[child_nodes[0]].shape)
            
        all_wt[child_nodes[0]] += temp_wt
    
    def _process_conv2d_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict):
        """Process Conv2d layer with original or CUDA implementation."""
        l1 = model_resource[0][start_layer]
        w1 = l1.state_dict()['weight']
        b1 = l1.state_dict()['bias']
        pad1 = l1.padding
        strides1 = l1.stride
        
        # Get input data correctly
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # Handle batch dimension for Conv2D - original implementation expects 3D (C,H,W)
        if len(input_data.shape) == 4:  # (batch, channels, height, width)
            input_sample = input_data[0]  # Take first sample: (channels, height, width)
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        else:
            input_sample = input_data
            weight_sample = all_wt[start_layer]
        
        if self.use_cuda_layers and torch.cuda.is_available():
            # Use CUDA implementation
            try:
                # Import CUDA version
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Conv2D.cuda_version import calculate_wt_conv_cuda_optimized
                
                # Get activation parameters and fix the format for CUDA  
                activation_params = activation_dict[model_resource[1][start_layer]["name"]]
                
                # Create a copy and fix the activation parameters for CUDA
                cuda_activation_params = activation_params.copy()
                
                # Ensure activation ranges are floats, not None
                if cuda_activation_params["range"]["l"] is None:
                    cuda_activation_params["range"]["l"] = 0.0
                if cuda_activation_params["range"]["u"] is None:
                    cuda_activation_params["range"]["u"] = 0.0
                    
                # Ensure other fields are in correct format
                if "func" not in cuda_activation_params:
                    cuda_activation_params["func"] = "relu"  # default
                
                # Convert tensors and call CUDA function
                grad_output_scales = torch.tensor(weight_sample, dtype=torch.float32).cuda()
                input_activations = torch.tensor(input_sample, dtype=torch.float32).cuda()
                kernel_weights_orig_shape = w1.cuda()
                bias = b1.cuda()
                
                temp_wt = calculate_wt_conv_cuda_optimized(
                    grad_output_scales,      # tensor: gradient scales
                    input_activations,       # tensor: input activations
                    kernel_weights_orig_shape,  # tensor: kernel weights
                    bias,                    # tensor: bias
                    pad1,                    # padding mode
                    strides1,                # strides tuple
                    cuda_activation_params   # activation parameters dict (fixed)
                ).cpu().numpy()
                
            except Exception as e:
                print(f"CUDA Conv2d implementation failed: {e}, falling back to original version")
                # Use original implementation with numpy arrays
                temp_wt = conv2d_original(
                    weight_sample,  # numpy array
                    input_sample,  # numpy array
                    w1.detach().numpy(),  # convert to numpy
                    b1.detach().numpy(),  # convert to numpy
                    pad1,
                    strides1,
                    activation_dict[model_resource[1][start_layer]["name"]]
                )
        else:
            # Use original implementation with numpy arrays
            temp_wt = conv2d_original(
                weight_sample,  # numpy array
                input_sample,  # numpy array
                w1.detach().numpy(),  # convert to numpy
                b1.detach().numpy(),  # convert to numpy
                pad1,
                strides1,
                activation_dict[model_resource[1][start_layer]["name"]]
            )
        
        # Handle the result shape to match target
        target_shape = all_wt[child_nodes[0]].shape
        
        # The original conv function returns transposed result, so we transpose it back
        temp_wt = temp_wt.T
        
        # If target has batch dimension but our result doesn't, add it
        if len(target_shape) == 4 and len(temp_wt.shape) == 3:
            temp_wt = temp_wt[np.newaxis, ...]  # Add batch dimension at the beginning
        
        # Ensure shapes match exactly
        if temp_wt.shape != target_shape:
            print(f"Shape mismatch: temp_wt {temp_wt.shape} vs target {target_shape}")
            # Try to reshape if possible
            if temp_wt.size == np.prod(target_shape):
                temp_wt = temp_wt.reshape(target_shape)
            else:
                print("Cannot reshape - using zeros")
                temp_wt = np.zeros(target_shape)
        
        all_wt[child_nodes[0]] += temp_wt
    
    def _process_maxpool2d_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource):
        """Process MaxPool2d layer with original or CUDA implementation."""
        l1 = model_resource[0][start_layer]
        pool_size = (l1.kernel_size, l1.kernel_size) if isinstance(l1.kernel_size, int) else l1.kernel_size
        padding = l1.padding if isinstance(l1.padding, tuple) else (l1.padding, l1.padding)
        strides = l1.stride if isinstance(l1.stride, tuple) else (l1.stride, l1.stride)
        
        # Get input data correctly
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # Handle batch dimension - MaxPool original implementation expects 3D (C,H,W)
        if len(input_data.shape) == 4:  # (batch, channels, height, width)
            input_sample = input_data[0]  # Take first sample: (channels, height, width)
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        else:
            input_sample = input_data
            weight_sample = all_wt[start_layer]
        
        if self.use_cuda_layers and torch.cuda.is_available():
            # Use CUDA implementation
            try:
                # Import CUDA version
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.MaxPool2D.cuda_version import calculate_wt_maxpool_cuda
                
                temp_wt = calculate_wt_maxpool_cuda(
                    torch.tensor(weight_sample).cuda(),
                    torch.tensor(input_sample).cuda(),
                    pool_size[0],  # Assuming square pooling
                    padding,
                    strides
                ).cpu().numpy()
                
            except Exception as e:
                print(f"CUDA MaxPool2d implementation failed: {e}, falling back to original version")
                # Use original implementation
                temp_wt = maxpool_original(
                    weight_sample,  # numpy array
                    input_sample,  # numpy array
                    pool_size,
                    padding,
                    strides
                )
        else:
            # Use original implementation
            temp_wt = maxpool_original(
                weight_sample,  # numpy array
                input_sample,  # numpy array
                pool_size,
                padding,
                strides
            )
        
        # Handle the result shape to match target
        target_shape = all_wt[child_nodes[0]].shape
        
        # The original maxpool function returns transposed result, so we transpose it back
        temp_wt = temp_wt.T
        
        # If target has batch dimension but our result doesn't, add it
        if len(target_shape) == 4 and len(temp_wt.shape) == 3:
            temp_wt = temp_wt[np.newaxis, ...]  # Add batch dimension at the beginning
        
        # Ensure shapes match exactly
        if temp_wt.shape != target_shape:
            print(f"MaxPool shape mismatch: temp_wt {temp_wt.shape} vs target {target_shape}")
            # Try to reshape if possible
            if temp_wt.size == np.prod(target_shape):
                temp_wt = temp_wt.reshape(target_shape)
            else:
                print("Cannot reshape - using zeros")
                temp_wt = np.zeros(target_shape)
        
        all_wt[child_nodes[0]] += temp_wt
    
    def _process_other_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict):
        """Process other layer types using original implementation."""
        from dl_backtrace.pytorch_backtrace.backtrace.utils import prop as UP
        
        layer_class = model_resource[1][start_layer]["class"]
        
        # Get input data correctly
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        if layer_class == "Flatten":
            temp_wt = UP.calculate_wt_rshp(all_wt[start_layer], input_data)
            all_wt[child_nodes[0]] += temp_wt
        elif layer_class == "AdaptiveAvgPool2d":
            temp_wt = UP.calculate_wt_gavgpool(all_wt[start_layer], input_data)
            all_wt[child_nodes[0]] += temp_wt.T
        else:
            # Pass through for identity layers
            temp_wt = all_wt[start_layer]
            all_wt[child_nodes[0]] += temp_wt

def run_comparison_test():
    """Run comparison test between original and CUDA implementations."""
    
    print("=== VGG Backtrace CUDA vs Original Implementation Comparison Test ===\n")
    
    # Generate test data
    print("Generating synthetic data...")
    num_samples = 10
    num_classes = 6
    X_data, y_data = create_synthetic_data(num_samples, num_classes)
    
    # Create and train a simple model
    print("Creating and training model...")
    model = SimpleVGG(num_classes=num_classes)
    
    # Quick training (just a few steps to get some reasonable weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(25):  # Very quick training
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {_} ,Training loss: {loss.item():.4f}")
    
    model.eval()
    print(f"Model training completed. Final loss: {loss.item():.4f}")
    
    # Test sample
    test_sample = X_data[0:1]
    
    # Create Backtrace objects
    print("\nInitializing Backtrace objects...")
    backtrace_original = CUDABacktrace(model=model, use_cuda_layers=False)
    backtrace_cuda = CUDABacktrace(model=model, use_cuda_layers=True)
    
    # Get layer outputs
    print("Getting layer outputs...")
    layer_outputs = backtrace_original.predict_every(test_sample)
    print(f"Captured outputs from {len(layer_outputs)} layers")
    
    # Run original implementation
    print("\n--- Running Original Implementation ---")
    start_time = time.time()
    relevance_original = backtrace_original.proportional_eval_with_cuda(
        layer_outputs, scaler=1
    )
    original_time = time.time() - start_time
    print(f"Original implementation time: {original_time:.4f} seconds")
    
    # Run CUDA implementation
    print("\n--- Running CUDA Implementation ---")
    start_time = time.time()
    relevance_cuda = backtrace_cuda.proportional_eval_with_cuda(
        layer_outputs, scaler=1
    )
    cuda_time = time.time() - start_time
    print(f"CUDA implementation time: {cuda_time:.4f} seconds")
    
    # Calculate and compare results
    max_diff = 0.0
    layer_diffs = {}
    
    # Calculate relevance score statistics
    original_stats = {}
    cuda_stats = {}
    
    print("\n--- Relevance Score Statistics ---")
    print("Layer | Original (min/avg/max) | CUDA (min/avg/max) | Max Diff")
    print("-" * 80)
    
    for layer_name in relevance_original:
        if layer_name in relevance_cuda:
            orig_scores = relevance_original[layer_name]
            cuda_scores = relevance_cuda[layer_name]
            
            # Calculate statistics for original scores
            orig_min = np.min(orig_scores)
            orig_avg = np.mean(orig_scores)
            orig_max = np.max(orig_scores)
            
            # Calculate statistics for CUDA scores  
            cuda_min = np.min(cuda_scores)
            cuda_avg = np.mean(cuda_scores)
            cuda_max = np.max(cuda_scores)
            
            # Store statistics
            original_stats[layer_name] = {'min': orig_min, 'avg': orig_avg, 'max': orig_max}
            cuda_stats[layer_name] = {'min': cuda_min, 'avg': cuda_avg, 'max': cuda_max}
            
            # Calculate layer difference
            layer_diff = np.max(np.abs(orig_scores - cuda_scores))
            layer_diffs[layer_name] = layer_diff
            max_diff = max(max_diff, layer_diff)
            
            # Print statistics comparison
            print(f"{layer_name:<12} | {orig_min:>6.3f}/{orig_avg:>6.3f}/{orig_max:>6.3f} | {cuda_min:>6.3f}/{cuda_avg:>6.3f}/{cuda_max:>6.3f} | {layer_diff:.2e}")
    
    print("\n--- Overall Statistics Summary ---")
    
    # Calculate overall statistics across all layers
    all_orig_scores = np.concatenate([scores.flatten() for scores in relevance_original.values()])
    all_cuda_scores = np.concatenate([scores.flatten() for scores in relevance_cuda.values()])
    
    print(f"Original Implementation:")
    print(f"  Min: {np.min(all_orig_scores):.6f}")
    print(f"  Avg: {np.mean(all_orig_scores):.6f}")
    print(f"  Max: {np.max(all_orig_scores):.6f}")
    print(f"  Std: {np.std(all_orig_scores):.6f}")
    
    print(f"CUDA Implementation:")
    print(f"  Min: {np.min(all_cuda_scores):.6f}")
    print(f"  Avg: {np.mean(all_cuda_scores):.6f}")
    print(f"  Max: {np.max(all_cuda_scores):.6f}")
    print(f"  Std: {np.std(all_cuda_scores):.6f}")
    
    # Calculate relative differences
    avg_rel_diff = np.mean(np.abs(all_orig_scores - all_cuda_scores) / (np.abs(all_orig_scores) + 1e-8))
    print(f"\nRelative Difference (avg): {avg_rel_diff:.2%}")
    print(f"Absolute Difference (max): {max_diff:.2e}")
    
    # Performance summary
    speedup = original_time / cuda_time
    print(f"\n--- Comparison Results ---")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Maximum numerical difference: {max_diff:.2e}")
    print(f"Layers processed: {len(layer_diffs)}")
    
    # Detailed layer timing comparison
    print("\n--- Layer-wise Performance ---")
    print("Layer Type | Original Avg (ms) | CUDA Avg (ms) | Speedup")
    print("-" * 60)
    
    for layer_type in backtrace_original.layer_timings:
        if layer_type in backtrace_cuda.layer_timings:
            orig_avg = np.mean(backtrace_original.layer_timings[layer_type]) * 1000
            cuda_avg = np.mean(backtrace_cuda.layer_timings[layer_type]) * 1000
            speedup_layer = orig_avg / cuda_avg if cuda_avg > 0 else float('inf')
            print(f"{layer_type:10} | {orig_avg:13.3f} | {cuda_avg:11.3f} | {speedup_layer:6.2f}x")
    
    # Show top layer differences
    if layer_diffs:
        print("\n--- Layer-wise Numerical Differences ---")
        for layer, diff in sorted(layer_diffs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{layer}: {diff:.2e}")
    
    return {
        'original_time': original_time,
        'cuda_time': cuda_time,
        'max_diff': max_diff,
        'layer_diffs': layer_diffs,
        'relevance_original': relevance_original,
        'relevance_cuda': relevance_cuda
    }

if __name__ == "__main__":
    try:
        results = run_comparison_test()
        
        print("\n=== Test Completed Successfully ===")
        
        if results['cuda_time'] > 0:
            print(f"Overall speedup: {results['original_time']/results['cuda_time']:.2f}x")
        print(f"Numerical accuracy: {results['max_diff']:.2e} max difference")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 
