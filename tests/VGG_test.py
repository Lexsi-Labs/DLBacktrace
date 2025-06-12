import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dl_backtrace.pytorch_backtrace.backtrace import backtrace as B
import graphviz
from IPython.display import Image, display
import os

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

# Generate synthetic data
num_samples = 1000
num_classes = 6
print("Generating synthetic data for testing...")
X_data, y_data = create_synthetic_data(num_samples, num_classes)

# Split into train and test
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size

X_train_tensor = X_data[:train_size]
Y_train_tensor = y_data[:train_size]
X_test_tensor = X_data[train_size:]
Y_test_tensor = y_data[train_size:]

print(f"Training samples: {train_size}, Test samples: {test_size}")
print(f"Number of classes: {num_classes}")

# Class mapping (for reference, though not strictly needed for synthetic data)
mapping = {f'class_{i}': i for i in range(num_classes)}

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
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

# Model setup
print("Initializing model...")
model = VGG19(num_classes=num_classes)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Move data to device
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training parameters
batch_size = 32  # Reduced for synthetic data
num_epochs = 5   # Reduced for testing purposes

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
print("Starting training...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    
    for batch_inputs, batch_labels in train_loader:
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

# Model evaluation
print("Evaluating model...")
model.eval()
with torch.no_grad():
    test_sample = X_test_tensor[0:1]  # Use first test sample
    test_output = model(test_sample)
    predicted_class = torch.argmax(test_output, dim=1).item()
    print(f"Test sample predicted class: {predicted_class}")

# Backtrace Analysis
print("Starting Backtrace analysis...")

# Move model back to CPU for backtrace (if it doesn't support GPU)
model = model.cpu()
test_sample = test_sample.cpu()

try:
    # Initialize Backtrace
    backtrace = B(model=model)

    # Get layer outputs
    print("Getting layer outputs...")
    layer_outputs = backtrace.predict_every(test_sample)
    print(f"Number of layers captured: {len(layer_outputs)}")
    
    # Calculate relevance
    print("Calculating relevance scores...")
    relevance = backtrace.eval(layer_outputs, mode='default', scaler=1)
    print(f"Relevance calculated for {len(relevance)} layers")
    
    # Process relevance data for visualization
    relevance_data = {}
    for layer, values in relevance.items():
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
    graph = graphviz.Digraph('Relevance_Tree', format='png')
    graph.attr(rankdir='TB', size='10,8')

    # Add nodes and edges
    layer_names = list(relevance_data.keys())
    for i, (layer, rel_score) in enumerate(relevance_data.items()):
        # Create node with layer name and relevance score
        label = f'{layer}\\nRelevance: {rel_score:.3f}'
        graph.node(layer, label=label, shape='box', style='filled', fillcolor='lightblue')
        
        # Add edge from previous layer (simple sequential connection)
    if i > 0:
            graph.edge(layer_names[i-1], layer)

    # Render the graph
    output_path = "relevance_tree"
    print(f"Rendering graph to {output_path}.png...")
    graph.render(output_path, format="png", cleanup=True)

    # Display results
    print("\nRelevance Summary:")
    print("-" * 40)
    for layer, score in relevance_data.items():
        print(f"{layer}: {score:.6f}")
    
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

print("VGG test completed!")
