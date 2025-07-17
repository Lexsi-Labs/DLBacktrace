import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dl_backtrace.pytorch_backtrace.backtrace.backtrace import Backtrace as B
import graphviz
from IPython.display import Image, display
import os
import time
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Create synthetic dataset for testing (replacing ImageNet)
def create_synthetic_data(num_samples=1000, num_classes=6, image_size=(3, 224, 224)):
    """
    Create synthetic image data for testing purposes.
    This avoids the need to download the large ImageNet dataset.
    """
    np.random.seed(42)  # For reproducible results
    torch.manual_seed(42)
    
    # Generate random image data
    X = torch.randn(num_samples, *image_size)
    # Normalize to [0, 1] range to simulate real images
    X = (X - X.min()) / (X.max() - X.min())
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

def initialize_binary_classification(root='./data'):
    """Initializes CIFAR10 for binary classification (cats vs dogs)."""
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)

    # --- Binary Classification Modification: Filter dataset for two classes ---
    binary_class_1 = 3  # cat
    binary_class_2 = 5  # dog
    classes = ('cat', 'dog')

    # Filter training data
    train_indices = [i for i, label in enumerate(trainset.targets) if label in [binary_class_1, binary_class_2]]
    trainset.data = trainset.data[train_indices]
    trainset.targets = [0 if trainset.targets[i] == binary_class_1 else 1 for i in train_indices]

    # Filter test data
    test_indices = [i for i, label in enumerate(testset.targets) if label in [binary_class_1, binary_class_2]]
    testset.data = testset.data[test_indices]
    testset.targets = [0 if testset.targets[i] == binary_class_1 else 1 for i in test_indices]

    num_classes = 2 # Set number of classes to 2 for binary task
    
    return trainset, testset, num_classes, classes

def initialize_multiclass_classification(root='./data'):
    """Initializes CIFAR10 for multi-class classification (all 10 classes)."""
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)
                                           
    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset, testset, num_classes, classes

# ResNet50 Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

# Custom ResNet50 Implementation
class ResNet50(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        
        self.identity = nn.Identity()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.identity(x)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def ResNet50_model(num_classes=10):
    """Create ResNet50 model with specified number of classes"""
    return ResNet50(Bottleneck, [3, 4, 6, 3], num_classes)

def load_pretrained_resnet50(num_classes):
    """
    Loads a pretrained ResNet-50 model and adapts it for the specified number of classes.
    This function uses torchvision's pretrained ResNet50 and modifies the final layer.
    """
    print("Loading pretrained ResNet-50 model from torchvision...")
    
    # Load pretrained ResNet50
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer for our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"Successfully loaded pretrained ResNet50 and adapted for {num_classes} classes.")
    
    return model

def test_model_performance(model, test_loader, device, classes, num_classes):
    """
    Evaluates model performance on the test set, providing overall accuracy,
    per-class accuracy, and a classification report.
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    all_preds = []
    all_labels = []
    
    total_inference_time = 0
    num_samples = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing Model Performance"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            num_samples += labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            c = (predicted == labels).squeeze()
            if len(c.shape) == 0: # Handle batch size of 1
                c = c.unsqueeze(0)

            for i in range(len(labels)):
                label = labels[i]
                if label < len(class_correct):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    print("\n=== Model Performance Results ===")
    accuracy = 100 * correct / total
    print(f'Overall Accuracy: {accuracy:.2f} %')
    
    avg_inference_time_ms = (total_inference_time / num_samples) * 1000
    print(f"Average inference time: {avg_inference_time_ms:.4f} ms per sample")

    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'  - {classes[i]:<10}: {100 * class_correct[i] / class_total[i]:.2f} % ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f'  - {classes[i]:<10}: N/A (no samples)')

    # Add classification report from scikit-learn
    try:
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        
        # Ensure that labels in classification_report are consistent
        unique_labels = sorted(list(set(all_labels)))
        target_names_filtered = [classes[i] for i in unique_labels]

        print(classification_report(all_labels, all_preds, target_names=target_names_filtered, digits=4, labels=unique_labels))
    except ImportError:
        print("\nScikit-learn not found. Skipping classification report.")
    except Exception as e:
        print(f"\nCould not generate classification report: {e}")
        
    return accuracy 

# Generate synthetic data
print("=== ResNet50 Backtrace CUDA Evaluation Benchmark ===")

# --- Choose Initialization ---
# Uncomment the desired initialization function
trainset, testset, num_classes, classes = initialize_multiclass_classification()
#trainset, testset, num_classes, classes = initialize_binary_classification()

print(f"Training samples: {len(trainset)}, Test samples: {len(testset)}")
print(f"Number of classes: {num_classes}")

# Class mapping for CIFAR10
mapping = {classes[i]: i for i in range(num_classes)}

# Model setup
print("\nInitializing ResNet50 model...")
# Choose between custom implementation or pretrained
# model = load_pretrained_resnet50(num_classes=num_classes)
# Use custom implementation which should work better with backtrace
model = ResNet50_model(num_classes=num_classes)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training parameters
batch_size = 64  # Slightly smaller for ResNet50 memory requirements
num_epochs = 2  # Reduced for testing purposes
num_workers = 2

# Create data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Training loop
print("\nStarting training...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_inputs, batch_labels in progress_bar:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

# Model performance testing
print("\n=== Model Performance Testing ===")
test_model_performance(model, test_loader, device, classes, num_classes)

# Model evaluation for Backtrace
print("\nSelecting sample for Backtrace analysis...")
model.eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    test_images, test_labels = next(dataiter)

    test_sample = test_images[0:1].to(device)  # Use first test sample from the batch
    test_output = model(test_sample)
    predicted_class = torch.argmax(test_output, dim=1).item()
    true_class = test_labels[0].item()
    print(f"Test sample predicted class: {predicted_class} (True Class: {true_class} - {classes[true_class]})")

# Backtrace Analysis
print("\n=== Backtrace Analysis and Benchmark ===")

# Move model back to CPU for backtrace (if it doesn't support GPU)
model = model.cpu()
test_sample = test_sample.cpu()

try:
    # Initialize Backtrace
    print("Initializing Backtrace...")
    backtrace = B(model=model)

    # Get layer outputs
    print("Getting layer outputs...")
    layer_outputs = backtrace.predict(test_sample)
    print(f"Number of layers captured: {len(layer_outputs)}")
    
    # Calculate relevance using default mode
    print("\n--- Running Default Mode ---")
    start_time = time.time()
    relevance_default = backtrace.eval(
        layer_outputs, 
        mode='default',
        scaler=0,
        task='multi-class classification', 
        multiplier=100.0
    )
    default_time = time.time() - start_time
    print(f"Default mode completed in {default_time:.4f} seconds")
    print(f"Relevance calculated for {len(relevance_default)} layers")
    
    # Calculate relevance using CUDA mode
    print("\n--- Running CUDA Eval Mode ---")
    start_time = time.time()
    relevance_cuda = backtrace.eval(
        layer_outputs, 
        mode='cuda_eval',
        scaler=0,
        task='multi-class classification', 
        multiplier=100.0
    )
    cuda_time = time.time() - start_time
    print(f"CUDA eval mode completed in {cuda_time:.4f} seconds")
    print(f"Relevance calculated for {len(relevance_cuda)} layers")
    
    # Performance comparison
    print(f"\n=== Performance Comparison ===")
    if cuda_time > 0:
        speedup = default_time / cuda_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("CUDA time too small to measure accurately")
    
    print(f"Default mode time: {default_time:.4f}s")
    print(f"CUDA eval mode time: {cuda_time:.4f}s")
    
    # Numerical accuracy comparison
    print(f"\n=== Numerical Accuracy Comparison ===")
    max_diff = 0.0
    layer_diffs = {}
    
    for layer_name in relevance_default:
        if layer_name in relevance_cuda:
            default_scores = relevance_default[layer_name]
            cuda_scores = relevance_cuda[layer_name]
            
            # Calculate layer difference
            layer_diff = np.max(np.abs(default_scores - cuda_scores))
            layer_diffs[layer_name] = layer_diff
            max_diff = max(max_diff, layer_diff)
    
    print(f"Maximum numerical difference: {max_diff:.2e}")
    print(f"Number of layers compared: {len(layer_diffs)}")
    
    # Show top differences
    if layer_diffs:
        print("\nTop 5 layer differences:")
        for layer, diff in sorted(layer_diffs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {layer}: {diff:.2e}")
    
    # Process relevance data for visualization - using CUDA results
    print(f"\n=== Creating Visualization (CUDA Results) ===")
    relevance_data = {}
    for layer, values in relevance_cuda.items():
        # Clean layer names for graphviz
        clean_layer_name = layer.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_')
        
        # Sum relevance values to get a single score per layer
        if isinstance(values, (np.ndarray, torch.Tensor)):
            if hasattr(values, 'cpu'):
                values = values.cpu()
            if hasattr(values, 'numpy'):
                values = values.numpy()
            relevance_score = float(np.sum(values.flatten()))
        else:
            relevance_score = float(values)
        
        relevance_data[clean_layer_name] = relevance_score
    
    print("Creating relevance visualization...")
    
    # Create directed graph
    graph = graphviz.Digraph('ResNet50_CUDA_Relevance_Tree', format='png')
    graph.attr(rankdir='TB', size='14,12')
    graph.attr('node', fontsize='9', width='1.0', height='0.6')

    # Add nodes and edges
    layer_names = list(relevance_data.keys())
    for i, (layer, rel_score) in enumerate(relevance_data.items()):
        # Create node with layer name and relevance score
        # Truncate long layer names for better visualization
        display_name = layer if len(layer) <= 15 else layer[:12] + "..."
        label = f'{display_name}\\nRel: {rel_score:.2f}'
        
        # Color coding based on relevance magnitude
        abs_score = abs(rel_score)
        if abs_score > 1000:
            color = 'red'      # High relevance
        elif abs_score > 100:
            color = 'orange'   # Medium relevance
        elif abs_score > 10:
            color = 'yellow'   # Low relevance
        else:
            color = 'lightblue'  # Very low relevance
            
        graph.node(layer, label=label, shape='box', style='filled', fillcolor=color)
        
        # Add edge from previous layer (simple sequential connection)
        if i > 0:
            graph.edge(layer_names[i-1], layer)

    # Render the graph
    output_path = "resnet50_cuda_relevance_benchmark"
    print(f"Rendering graph to {output_path}.png...")
    graph.render(output_path, format="png", cleanup=True)

    # Display results summary
    print(f"\n=== Final Benchmark Results ===")
    print("-" * 50)
    print(f"Model: ResNet50 with {num_classes} classes")
    print(f"Test sample class: {predicted_class}")
    print(f"Layers processed: {len(relevance_data)}")
    print(f"Default mode time: {default_time:.4f}s")
    print(f"CUDA eval mode time: {cuda_time:.4f}s")
    if cuda_time > 0:
        print(f"Performance improvement: {(default_time/cuda_time):.2f}x")
    print(f"Maximum numerical difference: {max_diff:.2e}")
    print("-" * 50)
    
    # Process relevance data for original results as well
    relevance_data_original = {}

    for layer, values in relevance_default.items():
        # Clean layer names for graphviz
        clean_layer_name = layer.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_')
        
        # Sum relevance values to get a single score per layer
        if isinstance(values, (np.ndarray, torch.Tensor)):
            if hasattr(values, 'cpu'):
                values = values.cpu()
            if hasattr(values, 'numpy'):
                values = values.numpy()
            relevance_score = float(np.sum(values.flatten()))
        else:
            relevance_score = float(values) 
        
        relevance_data_original[clean_layer_name] = relevance_score
    
    print("\nRelevance Summary (Original Implementation):")
    print("-" * 50)
    for layer, score in sorted(relevance_data_original.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"{layer:25}: {score:12.6f}")
    
    print("\nRelevance Summary (CUDA Implementation):")
    print("-" * 50)
    for layer, score in sorted(relevance_data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"{layer:25}: {score:12.6f}")
    
    print("\nStatistical Comparison of Relevance Scores:")
    print("-" * 70)
    
    # Get common layers
    common_layers = set(relevance_data_original.keys()) & set(relevance_data.keys())
    
    if common_layers:
        # Calculate statistics for original implementation
        orig_values = [relevance_data_original[layer] for layer in common_layers]
        orig_min = min(orig_values)
        orig_max = max(orig_values)
        orig_avg = sum(orig_values) / len(orig_values)
        orig_abs_values = [abs(val) for val in orig_values]
        orig_abs_min = min(orig_abs_values)
        orig_abs_max = max(orig_abs_values)
        orig_abs_avg = sum(orig_abs_values) / len(orig_abs_values)
        
        # Calculate statistics for CUDA implementation
        cuda_values = [relevance_data[layer] for layer in common_layers]
        cuda_min = min(cuda_values)
        cuda_max = max(cuda_values)
        cuda_avg = sum(cuda_values) / len(cuda_values)
        cuda_abs_values = [abs(val) for val in cuda_values]
        cuda_abs_min = min(cuda_abs_values)
        cuda_abs_max = max(cuda_abs_values)
        cuda_abs_avg = sum(cuda_abs_values) / len(cuda_abs_values)
        
        # Calculate differences
        differences = [abs(relevance_data_original[layer] - relevance_data[layer]) for layer in common_layers]
        diff_min = min(differences)
        diff_max = max(differences)
        diff_avg = sum(differences) / len(differences)
        
        print(f"{'Metric':<20} {'Original':<15} {'CUDA':<15} {'Abs Difference':<15}")
        print("-" * 70)
        print(f"{'Min Value':<20} {orig_min:<15.6f} {cuda_min:<15.6f} {diff_min:<15.2e}")
        print(f"{'Max Value':<20} {orig_max:<15.6f} {cuda_max:<15.6f} {diff_max:<15.2e}")
        print(f"{'Avg Value':<20} {orig_avg:<15.6f} {cuda_avg:<15.6f} {diff_avg:<15.2e}")
        print()
        print(f"{'Metric (Abs)':<20} {'Original':<15} {'CUDA':<15}")
        print("-" * 50)
        print(f"{'Min |Value|':<20} {orig_abs_min:<15.6f} {cuda_abs_min:<15.6f}")
        print(f"{'Max |Value|':<20} {orig_abs_max:<15.6f} {cuda_abs_max:<15.6f}")
        print(f"{'Avg |Value|':<20} {orig_abs_avg:<15.6f} {cuda_abs_avg:<15.6f}")
        print(f"{'Total Layers':<20} {len(common_layers):<15}")
    else:
        print("No common layers found for comparison.")
    
    # Try to display the image if in IPython environment
    try:
        image_path = f"{output_path}.png"
        if os.path.exists(image_path):
            display(Image(filename=image_path))
            print(f"\nVisualization saved and displayed: {image_path}")
        else:
            print(f"Image file not found: {image_path}")
    except Exception as e:
        print(f"Could not display image (might not be in IPython environment): {e}")
        print(f"Graph saved as: {output_path}.png")

except Exception as e:
    print(f"Error during Backtrace analysis: {e}")
    import traceback
    traceback.print_exc()

print("\nResNet50 CUDA benchmark test completed!") 
