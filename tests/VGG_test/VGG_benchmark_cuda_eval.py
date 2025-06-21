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

# Generate synthetic data
print("=== VGG Backtrace CUDA Evaluation Benchmark ===")
# print("Generating synthetic data for testing...")
# X_data, y_data = create_synthetic_data(num_samples, num_classes)

# --- Choose Initialization ---
# Uncomment the desired initialization function
trainset, testset, num_classes, classes = initialize_multiclass_classification()
#trainset, testset, num_classes, classes = initialize_binary_classification()


print(f"Training samples: {len(trainset)}, Test samples: {len(testset)}")
print(f"Number of classes: {num_classes}")

# Class mapping for CIFAR10
mapping = {classes[i]: i for i in range(num_classes)}

class VGG19(nn.Module):
    def __init__(self, num_classes=2): # Default to 2 for binary task
        super(VGG19, self).__init__()
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
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
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

def load_pretrained_vgg(num_classes):
    """
    Loads a pretrained VGG-19 model and maps its weights to the custom VGG19 model.
    This function transfers weights for all convolutional layers and the first two
    fully-connected layers to leverage the features learned from ImageNet.
    The final classification layer is left to be trained on the new dataset.
    """
    print("Loading pretrained VGG-19 model from torchvision...")
    pretrained_vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_vgg.state_dict()

    # Create an instance of the custom model
    custom_vgg = VGG19(num_classes=num_classes)
    custom_dict = custom_vgg.state_dict()

    # Create a detailed mapping from torchvision VGG19 layer names to custom model names
    mapping = {
        # Convolutional layers
        "features.0.weight": "conv1_1.weight", "features.0.bias": "conv1_1.bias",
        "features.2.weight": "conv1_2.weight", "features.2.bias": "conv1_2.bias",
        "features.5.weight": "conv2_1.weight", "features.5.bias": "conv2_1.bias",
        "features.7.weight": "conv2_2.weight", "features.7.bias": "conv2_2.bias",
        "features.10.weight": "conv3_1.weight", "features.10.bias": "conv3_1.bias",
        "features.12.weight": "conv3_2.weight", "features.12.bias": "conv3_2.bias",
        "features.14.weight": "conv3_3.weight", "features.14.bias": "conv3_3.bias",
        "features.16.weight": "conv3_4.weight", "features.16.bias": "conv3_4.bias",
        "features.19.weight": "conv4_1.weight", "features.19.bias": "conv4_1.bias",
        "features.21.weight": "conv4_2.weight", "features.21.bias": "conv4_2.bias",
        "features.23.weight": "conv4_3.weight", "features.23.bias": "conv4_3.bias",
        "features.25.weight": "conv4_4.weight", "features.25.bias": "conv4_4.bias",
        "features.28.weight": "conv5_1.weight", "features.28.bias": "conv5_1.bias",
        "features.30.weight": "conv5_2.weight", "features.30.bias": "conv5_2.bias",
        "features.32.weight": "conv5_3.weight", "features.32.bias": "conv5_3.bias",
        "features.34.weight": "conv5_4.weight", "features.34.bias": "conv5_4.bias",
        # Fully-connected layers (transferring first two)
        "classifier.0.weight": "fc1.weight", "classifier.0.bias": "fc1.bias",
        "classifier.3.weight": "fc2.weight", "classifier.3.bias": "fc2.bias",
    }

    # Create a new state dict for the custom model
    new_pretrained_dict = {}
    for torchvision_name, custom_name in mapping.items():
        if torchvision_name in pretrained_dict and custom_name in custom_dict:
            # Check if shapes are compatible before transferring
            if pretrained_dict[torchvision_name].shape == custom_dict[custom_name].shape:
                new_pretrained_dict[custom_name] = pretrained_dict[torchvision_name]
            else:
                print(f"Skipping layer {custom_name} due to shape mismatch.")
    
    # Update the custom model's dict and load the weights
    custom_dict.update(new_pretrained_dict)
    custom_vgg.load_state_dict(custom_dict, strict=False)
    print("Successfully loaded pretrained weights into the custom model.")
    
    return custom_vgg

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

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing Model Performance"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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

# Model setup
print("\nInitializing model...")
# Use VGGSmall for faster testing - change this line to switch between models
model = load_pretrained_vgg(num_classes=num_classes)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

# Data is now moved to the device in batches within the training loop.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training parameters
batch_size = 128  # Adjusted for CIFAR10
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
    
    for batch_inputs, batch_labels in train_loader:
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
        task='classification', 
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
        task='classification', 
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
        clean_layer_name = layer.replace('/', '_').replace(':', '_').replace(' ', '_')
        
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
    graph = graphviz.Digraph('CUDA_Relevance_Tree', format='png')
    graph.attr(rankdir='TB', size='12,10')
    graph.attr('node', fontsize='10', width='0.8', height='0.5')

    # Add nodes and edges
    layer_names = list(relevance_data.keys())
    for i, (layer, rel_score) in enumerate(relevance_data.items()):
        # Create node with layer name and relevance score
        label = f'{layer}\\nRelevance: {rel_score:.3f}'
        
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
    output_path = "cuda_relevance_benchmark"
    print(f"Rendering graph to {output_path}.png...")
    graph.render(output_path, format="png", cleanup=True)

    # Display results summary
    print(f"\n=== Final Benchmark Results ===")
    print("-" * 50)
    print(f"Model: VGG with {num_classes} classes")
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
        clean_layer_name = layer.replace('/', '_').replace(':', '_').replace(' ', '_')
        
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
        print(f"{layer:20}: {score:10.6f}")
    
    print("\nRelevance Summary (CUDA Implementation):")
    print("-" * 45)
    for layer, score in sorted(relevance_data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"{layer:20}: {score:10.6f}")
    
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

print("\nVGG CUDA benchmark test completed!") 


