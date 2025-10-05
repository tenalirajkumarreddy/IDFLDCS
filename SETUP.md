# Secure Federated Learning Setup Guide

## 🚀 Quick Setup

### 1. Environment Setup

```bash
# Create a virtual environment
python -m venv federated_env

# Activate virtual environment
# On Windows:
federated_env\Scripts\activate
# On macOS/Linux:
source federated_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Alternative: Install packages individually
pip install torch torchvision numpy
pip install cryptography pycryptodome
pip install matplotlib seaborn pandas
pip install scikit-learn
```

### 3. Verify Installation

```python
# Run this in Python to verify everything is working
import torch
import cryptography
import numpy as np
print("✅ All dependencies installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
```

## 🛠️ Detailed Installation Guide

### Prerequisites

1. **Python 3.8 or higher**
   - Download from [python.org](https://www.python.org/)
   - Verify: `python --version`

2. **Git** (optional, for cloning repository)
   - Download from [git-scm.com](https://git-scm.com/)

### Step-by-Step Installation

#### Windows Users:

```powershell
# 1. Clone or download the repository
git clone https://github.com/your-username/secure-federated-learning.git
cd secure-federated-learning

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test the installation
python secure_federated_learning.py
```

#### macOS/Linux Users:

```bash
# 1. Clone or download the repository
git clone https://github.com/your-username/secure-federated-learning.git
cd secure-federated-learning

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test the installation
python secure_federated_learning.py
```

### Common Installation Issues

#### Issue 1: PyTorch Installation Problems

```bash
# If PyTorch installation fails, try:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU support (optional):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 2: Cryptography Library Issues

```bash
# On older systems, you might need:
pip install --upgrade pip setuptools wheel
pip install cryptography

# On macOS with M1/M2 chips:
pip install cryptography --no-use-pep517
```

#### Issue 3: Visual C++ Requirements (Windows)

If you get build errors on Windows:
1. Install Microsoft Visual C++ Redistributable
2. Or install Microsoft Visual Studio Build Tools
3. Restart your terminal and try again

## 🧪 Testing Your Setup

### Basic Functionality Test

Create a file `test_setup.py`:

```python
"""Test script to verify your federated learning setup"""

def test_basic_imports():
    try:
        import torch
        import numpy as np
        from cryptography.hazmat.primitives import hashes
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_torch_functionality():
    try:
        # Test basic tensor operations
        x = torch.randn(5, 3)
        y = torch.randn(3, 5)
        z = torch.mm(x, y)
        print(f"✅ PyTorch test passed. Result shape: {z.shape}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_cryptography():
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        print("✅ Cryptography test passed")
        return True
    except Exception as e:
        print(f"❌ Cryptography error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Secure Federated Learning Setup...")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch Functionality", test_torch_functionality),
        ("Cryptography", test_cryptography)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Your setup is ready for secure federated learning!")
        print("Run 'python secure_federated_learning.py' to start the demo.")
    else:
        print("⚠️  Some tests failed. Please check your installation.")
```

Run the test:
```bash
python test_setup.py
```

### Advanced Test (Optional)

```python
"""Advanced test with actual federated learning simulation"""

def test_full_simulation():
    try:
        from secure_federated_learning import SecurityConfig, SecureFederatedServer
        import torch.nn as nn
        
        # Create simple model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # Test security configuration
        config = SecurityConfig(
            differential_privacy_epsilon=1.0,
            byzantine_tolerance=0.2
        )
        
        # Test server creation
        model = TestModel()
        server = SecureFederatedServer(model, config)
        
        print("✅ Advanced federated learning test passed")
        return True
        
    except Exception as e:
        print(f"❌ Advanced test error: {e}")
        return False

# Add this to the main test suite
```

## 🔧 Configuration Guide

### Security Configuration

Modify `SecurityConfig` in your code based on your needs:

```python
# High security setup (slower but more secure)
high_security_config = SecurityConfig(
    differential_privacy_epsilon=0.1,    # Strong privacy
    differential_privacy_delta=1e-7,     # Low delta
    byzantine_tolerance=0.4,             # High tolerance
    encryption_enabled=True,             # Always encrypt
    signature_verification=True,         # Always verify
    intrusion_detection_threshold=1.5    # Sensitive detection
)

# Balanced setup (recommended for most use cases)
balanced_config = SecurityConfig(
    differential_privacy_epsilon=1.0,
    differential_privacy_delta=1e-5,
    byzantine_tolerance=0.3,
    encryption_enabled=True,
    signature_verification=True,
    intrusion_detection_threshold=2.0
)

# Performance-focused setup (faster but less secure)
performance_config = SecurityConfig(
    differential_privacy_epsilon=5.0,    # Weaker privacy
    differential_privacy_delta=1e-3,
    byzantine_tolerance=0.2,
    encryption_enabled=False,            # No encryption
    signature_verification=False,        # No signatures
    intrusion_detection_threshold=3.0    # Less sensitive
)
```

### Hardware Requirements

**Minimum Requirements:**
- CPU: 2+ cores
- RAM: 4GB
- Storage: 1GB free space
- Python 3.8+

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 5GB+ free space
- GPU: Optional but recommended for larger models

**For Production Use:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ SSD
- GPU: NVIDIA GPU with CUDA support
- Network: Stable internet connection

## 🚀 Running Examples

### 1. Basic Demo
```bash
python secure_federated_learning.py
```

### 2. Advanced Cryptography Demo
```bash
python advanced_crypto.py
```

### 3. Attack Simulation
```bash
python attack_defense_simulation.py
```

### 4. Custom Configuration Example

Create `my_federated_learning.py`:

```python
from secure_federated_learning import *
import torch
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Configure security
my_config = SecurityConfig(
    differential_privacy_epsilon=1.5,
    byzantine_tolerance=0.25,
    encryption_enabled=True
)

# Create federated learning setup
model = MyModel(20, 50, 5)  # 20 input, 50 hidden, 5 output
server = SecureFederatedServer(model, my_config)

print("Custom federated learning setup ready!")
```

## 🔍 Troubleshooting

### Common Errors and Solutions

**Error:** `ModuleNotFoundError: No module named 'torch'`
**Solution:** 
```bash
pip install torch torchvision
```

**Error:** `ImportError: cannot import name 'default_backend'`
**Solution:**
```bash
pip install --upgrade cryptography
```

**Error:** `RuntimeError: CUDA out of memory`
**Solution:**
```python
# Use CPU instead of GPU
device = torch.device('cpu')
model = model.to(device)
```

**Error:** Permission denied during installation
**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Performance Optimization

```python
# For faster training (less secure)
fast_config = SecurityConfig(
    differential_privacy_epsilon=10.0,
    encryption_enabled=False,
    signature_verification=False
)

# For better security (slower)
secure_config = SecurityConfig(
    differential_privacy_epsilon=0.1,
    encryption_enabled=True,
    signature_verification=True
)
```

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** - Enable verbose logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Verify versions**:
   ```bash
   pip list | grep -E "(torch|cryptography|numpy)"
   ```

3. **Create a minimal reproduction case**

4. **Ask for help**:
   - GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
   - Discord Community: [Join our Discord](#)
   - Email: support@example.com

## 🎯 Next Steps

After successful setup:

1. **Read the main README.md** for detailed documentation
2. **Run the demos** to understand the framework
3. **Experiment with different configurations**
4. **Implement your own models and datasets**
5. **Contribute to the project**

**Happy Secure Federated Learning! 🚀🔒**