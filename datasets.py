"""
Dataset Management for Secure Federated Learning
This module provides utilities to load and prepare small datasets for federated learning experiments
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import urllib.request
import gzip
import pickle
from typing import List, Tuple, Dict

class FederatedDatasetManager:
    """Manages datasets for federated learning experiments"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def get_mnist_federated(self, num_clients=5, samples_per_client=1000, non_iid=False):
        """
        Load MNIST dataset and split it for federated learning
        
        Args:
            num_clients: Number of federated clients
            samples_per_client: Number of samples per client
            non_iid: Whether to create non-IID data distribution
        
        Returns:
            List of (train_loader, test_loader) tuples for each client
        """
        print(f"📥 Loading MNIST dataset for {num_clients} clients...")
        
        # Download and load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=transform
        )
        
        # Split training data among clients
        client_datasets = []
        
        if non_iid:
            # Create non-IID distribution (each client gets only 2-3 digit classes)
            client_datasets = self._create_non_iid_mnist(train_dataset, num_clients, samples_per_client)
        else:
            # Create IID distribution (random split)
            client_datasets = self._create_iid_split(train_dataset, num_clients, samples_per_client)
        
        # Create test loader (shared among all clients for evaluation)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create data loaders for each client
        federated_loaders = []
        for i, client_data in enumerate(client_datasets):
            train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
            federated_loaders.append((train_loader, test_loader))
            print(f"  Client {i+1}: {len(client_data)} training samples")
        
        return federated_loaders
    
    def get_cifar10_federated(self, num_clients=5, samples_per_client=2000):
        """Load CIFAR-10 dataset for federated learning"""
        print(f"📥 Loading CIFAR-10 dataset for {num_clients} clients...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform
        )
        
        # Split among clients
        client_datasets = self._create_iid_split(train_dataset, num_clients, samples_per_client)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        federated_loaders = []
        for i, client_data in enumerate(client_datasets):
            train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
            federated_loaders.append((train_loader, test_loader))
            print(f"  Client {i+1}: {len(client_data)} training samples")
        
        return federated_loaders
    
    def get_iris_federated(self, num_clients=3):
        """Load Iris dataset for federated learning (small dataset perfect for testing)"""
        print(f"📥 Loading Iris dataset for {num_clients} clients...")
        
        # Load Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Split among clients (each gets ~50 samples)
        samples_per_client = len(dataset) // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:  # Last client gets remaining samples
                end_idx = len(dataset)
            else:
                end_idx = (i + 1) * samples_per_client
            
            indices = list(range(start_idx, end_idx))
            client_data = torch.utils.data.Subset(dataset, indices)
            client_datasets.append(client_data)
        
        # Create loaders
        federated_loaders = []
        for i, client_data in enumerate(client_datasets):
            # Split into train/test for each client
            train_size = int(0.8 * len(client_data))
            test_size = len(client_data) - train_size
            train_data, test_data = random_split(client_data, [train_size, test_size])
            
            train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
            federated_loaders.append((train_loader, test_loader))
            print(f"  Client {i+1}: {len(train_data)} train, {len(test_data)} test samples")
        
        return federated_loaders
    
    def get_breast_cancer_federated(self, num_clients=4):
        """Load Breast Cancer dataset for federated learning"""
        print(f"📥 Loading Breast Cancer dataset for {num_clients} clients...")
        
        # Load dataset
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Split among clients
        samples_per_client = len(dataset) // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                end_idx = len(dataset)
            else:
                end_idx = (i + 1) * samples_per_client
            
            indices = list(range(start_idx, end_idx))
            client_data = torch.utils.data.Subset(dataset, indices)
            client_datasets.append(client_data)
        
        federated_loaders = []
        for i, client_data in enumerate(client_datasets):
            train_size = int(0.8 * len(client_data))
            test_size = len(client_data) - train_size
            train_data, test_data = random_split(client_data, [train_size, test_size])
            
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
            federated_loaders.append((train_loader, test_loader))
            print(f"  Client {i+1}: {len(train_data)} train, {len(test_data)} test samples")
        
        return federated_loaders
    
    def get_synthetic_federated(self, num_clients=5, samples_per_client=500, 
                               n_features=20, n_classes=3):
        """Generate synthetic dataset for federated learning"""
        print(f"📥 Generating synthetic dataset for {num_clients} clients...")
        
        federated_loaders = []
        
        for i in range(num_clients):
            # Generate synthetic data for each client
            X, y = make_classification(
                n_samples=samples_per_client,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=n_features//2,
                n_redundant=n_features//4,
                random_state=42 + i  # Different random state for each client
            )
            
            # Convert to tensors
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            
            # Create dataset
            dataset = torch.utils.data.TensorDataset(X, y)
            
            # Split into train/test
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_data, test_data = random_split(dataset, [train_size, test_size])
            
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
            federated_loaders.append((train_loader, test_loader))
            print(f"  Client {i+1}: {len(train_data)} train, {len(test_data)} test samples")
        
        return federated_loaders
    
    def _create_iid_split(self, dataset, num_clients, samples_per_client):
        """Create IID (Independent and Identically Distributed) split"""
        total_samples = min(len(dataset), num_clients * samples_per_client)
        
        # Random indices
        indices = torch.randperm(len(dataset))[:total_samples]
        
        client_datasets = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = min((i + 1) * samples_per_client, total_samples)
            client_indices = indices[start_idx:end_idx]
            client_data = torch.utils.data.Subset(dataset, client_indices)
            client_datasets.append(client_data)
        
        return client_datasets
    
    def _create_non_iid_mnist(self, dataset, num_clients, samples_per_client):
        """Create non-IID split for MNIST (each client gets only subset of classes)"""
        # Group data by class
        class_indices = {i: [] for i in range(10)}
        
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        
        client_datasets = []
        classes_per_client = 2  # Each client gets only 2 digit classes
        
        for i in range(num_clients):
            # Assign 2 classes to each client
            start_class = (i * classes_per_client) % 10
            client_classes = [start_class, (start_class + 1) % 10]
            
            client_indices = []
            samples_per_class = samples_per_client // len(client_classes)
            
            for class_id in client_classes:
                class_samples = class_indices[class_id][:samples_per_class]
                client_indices.extend(class_samples)
            
            client_data = torch.utils.data.Subset(dataset, client_indices)
            client_datasets.append(client_data)
        
        return client_datasets

# Model definitions for different datasets
class MNISTModel(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

class CIFAR10Model(nn.Module):
    """Simple CNN for CIFAR-10"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleMLPModel(nn.Module):
    """Simple MLP for tabular data (Iris, Breast Cancer)"""
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Dataset examples and usage
def demonstrate_datasets():
    """Demonstrate different datasets available for federated learning"""
    print("🔍 AVAILABLE DATASETS FOR SECURE FEDERATED LEARNING")
    print("=" * 60)
    
    dataset_manager = FederatedDatasetManager()
    
    print("\n1. 📋 SMALL DATASETS (Perfect for Testing):")
    print("   • Iris Dataset: 150 samples, 4 features, 3 classes")
    print("   • Breast Cancer: 569 samples, 30 features, 2 classes")
    print("   • Synthetic Data: Customizable size and complexity")
    
    print("\n2. 🖼️ IMAGE DATASETS:")
    print("   • MNIST: 60K training images, 28x28 handwritten digits")
    print("   • CIFAR-10: 50K training images, 32x32 color images, 10 classes")
    
    print("\n3. 🔬 RESEARCH DATASETS:")
    print("   • Fashion-MNIST: Clothing items (same format as MNIST)")
    print("   • EMNIST: Extended MNIST with letters")
    
    # Demonstrate loading small datasets
    print("\n" + "="*60)
    print("📥 LOADING DEMONSTRATION:")
    
    # 1. Iris dataset (very small, perfect for quick testing)
    print("\n--- Iris Dataset ---")
    iris_loaders = dataset_manager.get_iris_federated(num_clients=3)
    
    # 2. Breast Cancer dataset
    print("\n--- Breast Cancer Dataset ---")
    cancer_loaders = dataset_manager.get_breast_cancer_federated(num_clients=4)
    
    # 3. Synthetic dataset
    print("\n--- Synthetic Dataset ---")
    synthetic_loaders = dataset_manager.get_synthetic_federated(
        num_clients=5, samples_per_client=200, n_features=10, n_classes=2
    )
    
    return {
        'iris': iris_loaders,
        'cancer': cancer_loaders,
        'synthetic': synthetic_loaders
    }

if __name__ == "__main__":
    # Demonstrate available datasets
    datasets = demonstrate_datasets()
    
    print("\n" + "="*60)
    print("🚀 READY TO USE DATASETS:")
    print("="*60)
    
    print("\n📌 To use these datasets in your secure federated learning:")
    print("   1. Import: from datasets import FederatedDatasetManager")
    print("   2. Create manager: dm = FederatedDatasetManager()")
    print("   3. Load data: loaders = dm.get_iris_federated()")
    print("   4. Use in federated learning framework")
    
    print("\n🔧 Example Integration:")
    print("""
# Replace the dummy data in secure_federated_learning.py with:
from datasets import FederatedDatasetManager, SimpleMLPModel

# Load real dataset
dataset_manager = FederatedDatasetManager()
federated_loaders = dataset_manager.get_iris_federated(num_clients=5)

# Use appropriate model
model = SimpleMLPModel(input_size=4, num_classes=3)  # For Iris

# Create clients with real data
for i, (train_loader, test_loader) in enumerate(federated_loaders):
    client = SecureClient(f"client_{i}", model, train_loader, security_config)
    clients.append(client)
""")
    
    print("\n✅ All datasets are ready for secure federated learning!")
    print("📁 Data will be downloaded to: ./data/ directory")
    print("🔒 Compatible with all security mechanisms in the framework")