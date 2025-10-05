"""
🗄️ COMPREHENSIVE DATASET SOURCES FOR FEDERATED LEARNING RESEARCH
============================================================================

This guide provides sources for datasets suitable for federated learning experiments,
organized by category and complexity level.
"""

# ==============================================================================
# 🏁 QUICK START DATASETS (Built-in, No Download Required)
# ==============================================================================

QUICK_START_DATASETS = {
    "Iris": {
        "source": "scikit-learn.datasets.load_iris()",
        "size": "150 samples, 4 features, 3 classes",
        "type": "Tabular classification",
        "domain": "Botanical classification",
        "why_good_for_fl": "Small, simple, fast to test algorithms",
        "privacy_level": "Low (public botanical data)",
        "code_example": """
from sklearn.datasets import load_iris
from datasets import FederatedDatasetManager

# Load and split Iris dataset
manager = FederatedDatasetManager()
federated_data = manager.get_iris_federated(num_clients=3)
        """
    },
    
    "Breast Cancer Wisconsin": {
        "source": "scikit-learn.datasets.load_breast_cancer()",
        "size": "569 samples, 30 features, 2 classes",
        "type": "Tabular binary classification",
        "domain": "Medical diagnosis",
        "why_good_for_fl": "Medical data, realistic privacy concerns",
        "privacy_level": "High (medical data)",
        "code_example": """
from sklearn.datasets import load_breast_cancer
from datasets import FederatedDatasetManager

# Load and split breast cancer dataset
manager = FederatedDatasetManager()
federated_data = manager.get_breast_cancer_federated(num_clients=5)
        """
    },
    
    "Wine Quality": {
        "source": "scikit-learn.datasets.load_wine()",
        "size": "178 samples, 13 features, 3 classes",
        "type": "Tabular classification",
        "domain": "Food quality assessment",
        "why_good_for_fl": "Multi-class, real-world features",
        "privacy_level": "Medium (commercial data)",
        "code_example": """
from sklearn.datasets import load_wine
# Similar integration as other sklearn datasets
        """
    }
}

# ==============================================================================
# 🖼️ COMPUTER VISION DATASETS (Auto-downloadable)
# ==============================================================================

COMPUTER_VISION_DATASETS = {
    "MNIST": {
        "source": "torchvision.datasets.MNIST",
        "size": "60,000 training, 10,000 test images",
        "type": "Grayscale handwritten digits (28x28)",
        "domain": "Digit recognition",
        "why_good_for_fl": "Standard benchmark, well-studied",
        "privacy_level": "Low (handwritten digits)",
        "download_size": "~12 MB",
        "code_example": """
from datasets import FederatedDatasetManager

manager = FederatedDatasetManager()
federated_data = manager.get_mnist_federated(
    num_clients=10, 
    samples_per_client=1000,
    non_iid=True  # Realistic data distribution
)
        """
    },
    
    "CIFAR-10": {
        "source": "torchvision.datasets.CIFAR10",
        "size": "50,000 training, 10,000 test images",
        "type": "Color images (32x32x3)",
        "domain": "Object recognition (10 classes)",
        "why_good_for_fl": "More complex than MNIST, realistic images",
        "privacy_level": "Low-Medium (general objects)",
        "download_size": "~170 MB",
        "code_example": """
from datasets import FederatedDatasetManager

manager = FederatedDatasetManager()
federated_data = manager.get_cifar10_federated(
    num_clients=20,
    non_iid=True,
    alpha=0.5  # Controls non-IID level
)
        """
    },
    
    "Fashion-MNIST": {
        "source": "torchvision.datasets.FashionMNIST",
        "size": "60,000 training, 10,000 test images",
        "type": "Grayscale fashion items (28x28)",
        "domain": "Fashion item classification",
        "why_good_for_fl": "More challenging than MNIST, fashion domain",
        "privacy_level": "Low (fashion items)",
        "download_size": "~30 MB",
        "code_example": """
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# You can adapt the MNIST federated code for Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

fashion_mnist = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
        """
    }
}

# ==============================================================================
# 📊 MEDIUM-SIZE DATASETS (Manual Download Required)
# ==============================================================================

MEDIUM_DATASETS = {
    "UCI Adult (Census Income)": {
        "source": "https://archive.ics.uci.edu/ml/datasets/adult",
        "size": "48,842 samples, 14 features",
        "type": "Tabular binary classification",
        "domain": "Income prediction (>50K or <=50K)",
        "why_good_for_fl": "Demographic data, privacy-sensitive",
        "privacy_level": "High (personal income/demographics)",
        "download_size": "~4 MB",
        "code_example": """
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Download from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url, names=['age', 'workclass', 'fnlwgt', ...])

# Preprocess and create federated splits
# (You'll need to implement the preprocessing)
        """
    },
    
    "Credit Card Fraud Detection": {
        "source": "Kaggle - Credit Card Fraud Detection",
        "size": "284,807 transactions, 30 features",
        "type": "Tabular binary classification (fraud detection)",
        "domain": "Financial fraud detection",
        "why_good_for_fl": "Financial data, extreme class imbalance",
        "privacy_level": "Very High (financial transactions)",
        "download_size": "~144 MB",
        "requirements": "Kaggle account for download",
        "url": "https://www.kaggle.com/mlg-ulb/creditcardfraud"
    },
    
    "Human Activity Recognition": {
        "source": "UCI HAR Dataset",
        "size": "10,299 samples, 561 features",
        "type": "Time series classification (6 activities)",
        "domain": "Smartphone sensor data",
        "why_good_for_fl": "Personal sensor data, IoT scenario",
        "privacy_level": "High (personal activity data)",
        "download_size": "~60 MB",
        "url": "https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones"
    }
}

# ==============================================================================
# 🏥 SPECIALIZED FEDERATED LEARNING DATASETS
# ==============================================================================

FEDERATED_SPECIFIC_DATASETS = {
    "LEAF Benchmark": {
        "source": "https://leaf.cmu.edu/",
        "description": "Benchmark specifically designed for federated learning",
        "datasets": {
            "FEMNIST": "Federated Extended MNIST (handwriting from different users)",
            "Sent140": "Sentiment analysis from Twitter (by user)",
            "Shakespeare": "Text generation from Shakespeare plays (by character)",
            "Celeba": "Celebrity faces (by celebrity)",
            "Synthetic": "Generated federated datasets with controllable properties"
        },
        "why_special": "Natural federated partitioning (by user/device)",
        "privacy_level": "Varies by dataset",
        "url": "https://github.com/TalwalkarLab/leaf"
    },
    
    "FedML Datasets": {
        "source": "https://fedml.ai/",
        "description": "Curated datasets for federated learning research",
        "datasets": {
            "Fed-CIFAR100": "CIFAR-100 with federated splits",
            "Fed-Shakespeare": "Text generation with natural federated splits",
            "Fed-StackOverflow": "Next word prediction from StackOverflow"
        },
        "why_special": "Pre-processed for federated learning",
        "url": "https://github.com/FedML-AI/FedML"
    }
}

# ==============================================================================
# 🧬 DOMAIN-SPECIFIC DATASETS
# ==============================================================================

DOMAIN_SPECIFIC_DATASETS = {
    "Medical/Healthcare": {
        "MIMIC-III": {
            "description": "ICU patient data",
            "size": "Large (requires application)",
            "privacy": "Very High",
            "url": "https://mimic.physionet.org/"
        },
        "eICU": {
            "description": "Multi-center ICU data",
            "size": "Large",
            "privacy": "Very High",
            "url": "https://eicu-crd.mit.edu/"
        },
        "Chest X-Ray": {
            "description": "Medical imaging",
            "size": "100,000+ images",
            "privacy": "High",
            "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC"
        }
    },
    
    "Financial": {
        "Stock Market Data": {
            "description": "Historical stock prices",
            "sources": ["Yahoo Finance", "Alpha Vantage", "Quandl"],
            "privacy": "Medium",
            "why_federated": "Different markets/institutions"
        },
        "Cryptocurrency": {
            "description": "Crypto trading data",
            "sources": ["Coinbase API", "Binance API"],
            "privacy": "Medium",
            "why_federated": "Different exchanges"
        }
    },
    
    "IoT/Sensor Data": {
        "Environmental Sensing": {
            "description": "Temperature, humidity, air quality",
            "sources": ["OpenWeatherMap", "PurpleAir"],
            "privacy": "Low-Medium",
            "why_federated": "Distributed sensors"
        },
        "Smart Home": {
            "description": "Home automation data",
            "privacy": "High",
            "why_federated": "Different households"
        }
    }
}

# ==============================================================================
# 🔧 HOW TO USE THESE DATASETS
# ==============================================================================

INTEGRATION_GUIDE = """
📋 STEP-BY-STEP INTEGRATION GUIDE:

1. 🎯 Choose Your Dataset:
   - Start with Iris/Breast Cancer for testing
   - Move to MNIST/CIFAR-10 for computer vision
   - Use specialized datasets for your research domain

2. 📥 Download/Load Data:
   - Built-in: Use scikit-learn or torchvision
   - Manual: Download from provided URLs
   - Kaggle: Create account and use Kaggle API

3. 🔄 Adapt to Federated Setting:
   - Modify FederatedDatasetManager in datasets.py
   - Create appropriate train/test splits per client
   - Consider IID vs non-IID data distribution

4. 🏗️ Create Model Architecture:
   - Tabular data: Use SimpleMLPModel as template
   - Images: Use MNISTModel or CIFAR10Model as template
   - Custom: Create PyTorch nn.Module

5. 🔐 Configure Security:
   - Adjust SecurityConfig parameters
   - Consider privacy level of your dataset
   - Set appropriate differential privacy epsilon

6. 🚀 Run Experiments:
   - Use dataset_examples.py as template
   - Modify for your specific dataset
   - Monitor security metrics
"""

def print_dataset_recommendations():
    """Print personalized dataset recommendations"""
    print("🎯 DATASET RECOMMENDATIONS BY USE CASE:")
    print("="*50)
    
    recommendations = {
        "🧪 Learning/Testing": ["Iris", "Breast Cancer", "Wine"],
        "🖼️ Computer Vision": ["MNIST", "CIFAR-10", "Fashion-MNIST"],
        "🏥 Medical Research": ["Breast Cancer", "MIMIC-III", "Chest X-Ray"],
        "💰 Financial": ["Credit Card Fraud", "Stock Market", "Adult Income"],
        "📱 IoT/Mobile": ["Human Activity Recognition", "Environmental Sensors"],
        "📚 Natural Language": ["Shakespeare", "Sent140", "StackOverflow"],
        "🔬 Research Benchmarks": ["LEAF datasets", "FedML datasets"]
    }
    
    for use_case, datasets in recommendations.items():
        print(f"\n{use_case}:")
        for dataset in datasets:
            print(f"  • {dataset}")

def print_privacy_considerations():
    """Print privacy considerations for different datasets"""
    print("\n🔒 PRIVACY CONSIDERATIONS:")
    print("="*50)
    
    privacy_levels = {
        "🟢 Low Privacy Risk": {
            "datasets": ["Iris", "MNIST", "CIFAR-10", "Wine"],
            "epsilon_range": "ε = 5-10 (less noise needed)",
            "considerations": "Public data, less sensitive"
        },
        "🟡 Medium Privacy Risk": {
            "datasets": ["Adult Income", "Stock Market", "Environmental"],
            "epsilon_range": "ε = 1-5 (moderate noise)",
            "considerations": "Semi-sensitive, demographic/financial"
        },
        "🔴 High Privacy Risk": {
            "datasets": ["Breast Cancer", "Credit Card Fraud", "Activity Recognition"],
            "epsilon_range": "ε = 0.1-1 (strong noise)",
            "considerations": "Medical, financial, personal behavior"
        },
        "🟥 Very High Privacy Risk": {
            "datasets": ["MIMIC-III", "Smart Home", "Personal Communications"],
            "epsilon_range": "ε = 0.01-0.1 (very strong noise)",
            "considerations": "Highly sensitive, requires strong protection"
        }
    }
    
    for level, info in privacy_levels.items():
        print(f"\n{level}:")
        print(f"  Datasets: {', '.join(info['datasets'])}")
        print(f"  Recommended: {info['epsilon_range']}")
        print(f"  Notes: {info['considerations']}")

def main():
    """Display comprehensive dataset guide"""
    print(__doc__)
    
    print_dataset_recommendations()
    print_privacy_considerations()
    
    print("\n" + "="*70)
    print("🚀 QUICK START COMMAND:")
    print("="*70)
    print("python dataset_examples.py  # Run examples with different datasets")
    print("\n📁 All datasets will be downloaded to './data/' folder")
    print("💡 Start with Iris dataset for quick testing!")

if __name__ == "__main__":
    main()