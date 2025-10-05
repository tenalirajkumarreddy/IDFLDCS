"""
Example: Secure Federated Learning with Real Datasets
This script demonstrates how to use real datasets with the secure federated learning framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import FederatedDatasetManager, MNISTModel, SimpleMLPModel, CIFAR10Model
from secure_federated_learning import SecurityConfig, SecureClient, SecureFederatedServer
import torch
import torch.nn as nn

def run_federated_learning_with_iris():
    """Run secure federated learning with Iris dataset"""
    print("🌸 SECURE FEDERATED LEARNING WITH IRIS DATASET")
    print("=" * 60)
    
    # Load Iris dataset
    dataset_manager = FederatedDatasetManager()
    federated_loaders = dataset_manager.get_iris_federated(num_clients=3)
    
    # Security configuration
    security_config = SecurityConfig(
        differential_privacy_epsilon=2.0,  # Moderate privacy for small dataset
        byzantine_tolerance=0.2,           # Allow 1 malicious client out of 3
        encryption_enabled=True,
        signature_verification=True,
        intrusion_detection_threshold=2.5
    )
    
    # Create appropriate model for Iris (4 features, 3 classes)
    global_model = SimpleMLPModel(input_size=4, hidden_size=32, num_classes=3)
    
    # Initialize secure server
    server = SecureFederatedServer(global_model, security_config)
    
    # Create secure clients with real data
    clients = []
    for i, (train_loader, test_loader) in enumerate(federated_loaders):
        client_model = SimpleMLPModel(input_size=4, hidden_size=32, num_classes=3)
        client = SecureClient(f"iris_client_{i}", client_model, train_loader, security_config)
        clients.append(client)
        
        # Register client with server
        server.register_client(client.client_id, client.get_public_key_pem())
        print(f"✅ Registered {client.client_id}")
    
    # Run federated learning rounds
    num_rounds = 5
    print(f"\n🔄 Starting {num_rounds} federated learning rounds...")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Collect updates from clients
        client_updates = {}
        for client in clients:
            update = client.local_training(epochs=2)
            signature = client.sign_update(update)
            client_updates[client.client_id] = (update, signature)
            print(f"📤 Received update from {client.client_id}")
        
        # Server aggregates updates securely
        aggregated_update = server.aggregate_updates(client_updates)
        server.update_global_model(aggregated_update)
        print(f"🔒 Secure aggregation completed for round {round_num + 1}")
    
    print("\n✅ Iris federated learning completed successfully!")
    return server, clients

def run_federated_learning_with_mnist():
    """Run secure federated learning with MNIST dataset"""
    print("\n🔢 SECURE FEDERATED LEARNING WITH MNIST DATASET")
    print("=" * 60)
    
    # Load MNIST dataset (smaller subset for demo)
    dataset_manager = FederatedDatasetManager()
    federated_loaders = dataset_manager.get_mnist_federated(
        num_clients=5, 
        samples_per_client=500,  # Small subset for demo
        non_iid=True  # Make it more realistic
    )
    
    # Security configuration for larger dataset
    security_config = SecurityConfig(
        differential_privacy_epsilon=1.0,  # Stronger privacy for image data
        byzantine_tolerance=0.3,           # Can handle 1-2 malicious clients
        encryption_enabled=True,
        signature_verification=True,
        intrusion_detection_threshold=2.0
    )
    
    # Create CNN model for MNIST
    global_model = MNISTModel()
    server = SecureFederatedServer(global_model, security_config)
    
    # Create clients with MNIST data
    clients = []
    for i, (train_loader, test_loader) in enumerate(federated_loaders):
        client_model = MNISTModel()
        client = SecureClient(f"mnist_client_{i}", client_model, train_loader, security_config)
        clients.append(client)
        server.register_client(client.client_id, client.get_public_key_pem())
        print(f"✅ Registered {client.client_id}")
    
    # Run fewer rounds for demo
    num_rounds = 3
    print(f"\n🔄 Starting {num_rounds} federated learning rounds...")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        client_updates = {}
        for client in clients:
            update = client.local_training(epochs=1)  # Quick training
            signature = client.sign_update(update)
            client_updates[client.client_id] = (update, signature)
            print(f"📤 Received update from {client.client_id}")
        
        aggregated_update = server.aggregate_updates(client_updates)
        server.update_global_model(aggregated_update)
        print(f"🔒 Secure aggregation completed for round {round_num + 1}")
    
    print("\n✅ MNIST federated learning completed successfully!")
    return server, clients

def run_federated_learning_with_synthetic():
    """Run secure federated learning with synthetic dataset"""
    print("\n🧪 SECURE FEDERATED LEARNING WITH SYNTHETIC DATASET")
    print("=" * 60)
    
    # Generate synthetic dataset
    dataset_manager = FederatedDatasetManager()
    federated_loaders = dataset_manager.get_synthetic_federated(
        num_clients=4,
        samples_per_client=300,
        n_features=15,
        n_classes=5
    )
    
    # Security configuration
    security_config = SecurityConfig(
        differential_privacy_epsilon=1.5,
        byzantine_tolerance=0.25,
        encryption_enabled=True,
        signature_verification=True,
        intrusion_detection_threshold=2.0
    )
    
    # Create model for synthetic data
    global_model = SimpleMLPModel(input_size=15, hidden_size=64, num_classes=5)
    server = SecureFederatedServer(global_model, security_config)
    
    # Create clients
    clients = []
    for i, (train_loader, test_loader) in enumerate(federated_loaders):
        client_model = SimpleMLPModel(input_size=15, hidden_size=64, num_classes=5)
        client = SecureClient(f"synthetic_client_{i}", client_model, train_loader, security_config)
        clients.append(client)
        server.register_client(client.client_id, client.get_public_key_pem())
        print(f"✅ Registered {client.client_id}")
    
    # Run federated learning
    num_rounds = 4
    print(f"\n🔄 Starting {num_rounds} federated learning rounds...")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        client_updates = {}
        for client in clients:
            update = client.local_training(epochs=2)
            signature = client.sign_update(update)
            client_updates[client.client_id] = (update, signature)
            print(f"📤 Received update from {client.client_id}")
        
        aggregated_update = server.aggregate_updates(client_updates)
        server.update_global_model(aggregated_update)
        print(f"🔒 Secure aggregation completed for round {round_num + 1}")
    
    print("\n✅ Synthetic data federated learning completed successfully!")
    return server, clients

def compare_datasets():
    """Compare different datasets for federated learning"""
    print("\n📊 DATASET COMPARISON FOR FEDERATED LEARNING")
    print("=" * 70)
    
    datasets_info = [
        {
            'name': 'Iris',
            'samples': 150,
            'features': 4,
            'classes': 3,
            'type': 'Tabular',
            'size': 'Very Small',
            'use_case': 'Quick testing, algorithm validation',
            'security_considerations': 'High privacy risk due to small size'
        },
        {
            'name': 'Breast Cancer',
            'samples': 569,
            'features': 30,
            'classes': 2,
            'type': 'Tabular',
            'size': 'Small',
            'use_case': 'Medical research, binary classification',
            'security_considerations': 'Medical data requires strong privacy'
        },
        {
            'name': 'MNIST',
            'samples': 60000,
            'features': 784,
            'classes': 10,
            'type': 'Image',
            'size': 'Medium',
            'use_case': 'Computer vision, deep learning',
            'security_considerations': 'Gradient leakage attacks possible'
        },
        {
            'name': 'CIFAR-10',
            'samples': 50000,
            'features': 3072,
            'classes': 10,
            'type': 'Image',
            'size': 'Medium',
            'use_case': 'Complex computer vision tasks',
            'security_considerations': 'Model inversion attacks possible'
        },
        {
            'name': 'Synthetic',
            'samples': 'Customizable',
            'features': 'Customizable',
            'classes': 'Customizable',
            'type': 'Generated',
            'size': 'Any',
            'use_case': 'Controlled experiments, research',
            'security_considerations': 'No real privacy concerns'
        }
    ]
    
    for dataset in datasets_info:
        print(f"\n📋 {dataset['name']} Dataset:")
        print(f"   Samples: {dataset['samples']}")
        print(f"   Features: {dataset['features']}")
        print(f"   Classes: {dataset['classes']}")
        print(f"   Type: {dataset['type']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Use Case: {dataset['use_case']}")
        print(f"   Security: {dataset['security_considerations']}")

def main():
    """Run demonstrations with different datasets"""
    print("🚀 SECURE FEDERATED LEARNING WITH REAL DATASETS")
    print("=" * 70)
    
    # Show available datasets
    compare_datasets()
    
    # Run examples (comment out the ones you don't want to run)
    
    print("\n" + "="*70)
    print("🔬 RUNNING DEMONSTRATIONS...")
    print("="*70)
    
    # 1. Run with Iris dataset (fastest, good for testing)
    try:
        iris_server, iris_clients = run_federated_learning_with_iris()
        print("✅ Iris demonstration completed")
    except Exception as e:
        print(f"❌ Iris demonstration failed: {e}")
    
    # 2. Run with synthetic dataset (customizable)
    try:
        synthetic_server, synthetic_clients = run_federated_learning_with_synthetic()
        print("✅ Synthetic demonstration completed")
    except Exception as e:
        print(f"❌ Synthetic demonstration failed: {e}")
    
    # 3. Run with MNIST (comment out if you want to skip - takes longer)
    # try:
    #     mnist_server, mnist_clients = run_federated_learning_with_mnist()
    #     print("✅ MNIST demonstration completed")
    # except Exception as e:
    #     print(f"❌ MNIST demonstration failed: {e}")
    
    print("\n🎉 ALL DEMONSTRATIONS COMPLETED!")
    print("\n📁 Where to find datasets:")
    print("   • Iris & Breast Cancer: Built into scikit-learn")
    print("   • MNIST & CIFAR-10: Auto-downloaded to ./data/ folder")
    print("   • Synthetic: Generated on-the-fly")
    
    print("\n🔧 How to add your own dataset:")
    print("   1. Create a PyTorch Dataset class")
    print("   2. Split it among clients using FederatedDatasetManager")
    print("   3. Create appropriate model architecture")
    print("   4. Integrate with SecureClient and SecureFederatedServer")

if __name__ == "__main__":
    main()