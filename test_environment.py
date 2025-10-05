#!/usr/bin/env python3
"""
Test script to verify the secure federated learning environment setup
"""

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - SUCCESS")
    except ImportError as e:
        print(f"❌ PyTorch - FAILED: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} - SUCCESS")
    except ImportError as e:
        print(f"❌ NumPy - FAILED: {e}")
        return False
    
    try:
        import cryptography
        print(f"✅ Cryptography {cryptography.__version__} - SUCCESS")
    except ImportError as e:
        print(f"❌ Cryptography - FAILED: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} - SUCCESS")
    except ImportError as e:
        print(f"❌ Matplotlib - FAILED: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} - SUCCESS")
    except ImportError as e:
        print(f"❌ Pandas - FAILED: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} - SUCCESS")
    except ImportError as e:
        print(f"❌ Scikit-learn - FAILED: {e}")
        return False
    
    return True

def test_torch_functionality():
    """Test basic PyTorch operations"""
    print("\n🔥 Testing PyTorch functionality...")
    
    try:
        import torch
        
        # Test tensor creation
        x = torch.randn(3, 4)
        y = torch.randn(4, 3)
        z = torch.mm(x, y)
        
        print(f"✅ Tensor operations - Shape: {z.shape}")
        
        # Test neural network
        import torch.nn as nn
        model = nn.Linear(10, 5)
        input_data = torch.randn(2, 10)
        output = model(input_data)
        
        print(f"✅ Neural network - Output shape: {output.shape}")
        
        # Check if CUDA is available (optional)
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available - using CPU (this is fine)")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        return False

def test_cryptography():
    """Test cryptography functionality"""
    print("\n🔐 Testing cryptography functionality...")
    
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        # Generate key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        
        # Test encryption/decryption
        message = b"Hello, Secure Federated Learning!"
        
        encrypted = public_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        decrypted = private_key.decrypt(
            encrypted,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        if decrypted == message:
            print("✅ RSA encryption/decryption - SUCCESS")
            return True
        else:
            print("❌ RSA encryption/decryption - FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Cryptography test failed: {e}")
        return False

def test_federated_learning_basics():
    """Test basic federated learning functionality"""
    print("\n🤝 Testing federated learning basics...")
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # Create multiple client models
        num_clients = 3
        models = [SimpleModel() for _ in range(num_clients)]
        
        # Simulate federated averaging
        print(f"✅ Created {num_clients} client models")
        
        # Get model parameters
        client_params = []
        for model in models:
            params = []
            for param in model.parameters():
                params.append(param.data.flatten())
            client_params.append(torch.cat(params))
        
        # Average parameters (basic federated averaging)
        averaged_params = torch.stack(client_params).mean(dim=0)
        
        print(f"✅ Federated averaging - Parameter vector size: {len(averaged_params)}")
        
        # Test differential privacy simulation
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 1.0
        
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = torch.normal(0, sigma, averaged_params.shape)
        private_params = averaged_params + noise
        
        privacy_cost = torch.norm(noise) / torch.norm(averaged_params)
        print(f"✅ Differential privacy simulation - Privacy cost: {privacy_cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Federated learning test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔒 SECURE FEDERATED LEARNING ENVIRONMENT TEST")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch Functionality", test_torch_functionality),
        ("Cryptography", test_cryptography),
        ("Federated Learning Basics", test_federated_learning_basics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} - CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your environment is ready for secure federated learning!")
        print("\nNext steps:")
        print("1. Run: python secure_federated_learning.py")
        print("2. Run: python implementation_guide.py")
        print("3. Explore: python advanced_crypto.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check your installation.")
    
    return passed == total

if __name__ == "__main__":
    main()