"""
Attack Scenarios and Defense Mechanisms in Federated Learning
This module demonstrates various attacks and their corresponding defenses
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
from secure_federated_learning import (
    SecureClient, SecureFederatedServer, SecurityConfig,
    ByzantineRobustAggregation, IntrusionDetector, DifferentialPrivacy
)

@dataclass
class AttackConfig:
    """Configuration for different attack scenarios"""
    attack_type: str
    num_malicious_clients: int
    attack_intensity: float
    target_accuracy_drop: float = 0.1

class AttackSimulator:
    """Simulates various attack scenarios in federated learning"""
    
    def __init__(self, attack_config: AttackConfig):
        self.attack_config = attack_config
        self.attack_history = []
    
    def model_poisoning_attack(self, clean_update: torch.Tensor) -> torch.Tensor:
        """
        Simulate model poisoning attack by corrupting model updates
        
        Strategies:
        1. Random noise injection
        2. Sign flipping
        3. Scaling attacks
        """
        poisoned_update = clean_update.clone()
        
        if self.attack_config.attack_type == "random_noise":
            # Add large random noise
            noise = torch.randn_like(clean_update) * self.attack_config.attack_intensity
            poisoned_update += noise
            
        elif self.attack_config.attack_type == "sign_flipping":
            # Flip the sign of updates (opposite direction)
            poisoned_update = -clean_update * self.attack_config.attack_intensity
            
        elif self.attack_config.attack_type == "scaling_attack":
            # Scale updates by large factor
            poisoned_update = clean_update * self.attack_config.attack_intensity
            
        elif self.attack_config.attack_type == "targeted_poisoning":
            # Target specific parameters (e.g., bias terms)
            # This is a simplified version - real attacks are more sophisticated
            with torch.no_grad():
                poisoned_update += torch.randn_like(clean_update) * 0.1
                poisoned_update[:100] *= self.attack_config.attack_intensity  # Target first 100 params
        
        self.attack_history.append({
            'attack_type': self.attack_config.attack_type,
            'intensity': self.attack_config.attack_intensity,
            'norm_difference': torch.norm(poisoned_update - clean_update).item()
        })
        
        return poisoned_update
    
    def data_poisoning_attack(self, data_loader, poison_ratio: float = 0.1):
        """
        Simulate data poisoning by corrupting training labels
        """
        poisoned_data = []
        poisoned_labels = []
        
        for data, labels in data_loader:
            # Randomly flip labels for poison_ratio of samples
            num_samples = len(labels)
            num_poisoned = int(num_samples * poison_ratio)
            poison_indices = random.sample(range(num_samples), num_poisoned)
            
            poisoned_batch_labels = labels.clone()
            for idx in poison_indices:
                # Flip to random wrong label
                current_label = labels[idx].item()
                wrong_labels = [i for i in range(10) if i != current_label]  # Assuming 10 classes
                poisoned_batch_labels[idx] = random.choice(wrong_labels)
            
            poisoned_data.append(data)
            poisoned_labels.append(poisoned_batch_labels)
        
        return list(zip(poisoned_data, poisoned_labels))
    
    def gradient_leakage_simulation(self, gradients: torch.Tensor) -> Dict:
        """
        Simulate gradient leakage attack (simplified version)
        
        In reality, this would attempt to reconstruct training data from gradients
        """
        # Analyze gradient statistics that might leak information
        grad_stats = {
            'mean': torch.mean(gradients).item(),
            'std': torch.std(gradients).item(),
            'min': torch.min(gradients).item(),
            'max': torch.max(gradients).item(),
            'norm': torch.norm(gradients).item()
        }
        
        # Simulate information leakage score (higher = more leakage)
        leakage_score = abs(grad_stats['mean']) + grad_stats['std']
        
        return {
            'statistics': grad_stats,
            'leakage_score': leakage_score,
            'privacy_risk': 'HIGH' if leakage_score > 1.0 else 'MEDIUM' if leakage_score > 0.5 else 'LOW'
        }

class DefenseEvaluator:
    """Evaluates the effectiveness of different defense mechanisms"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_differential_privacy(self, clean_updates: List[torch.Tensor], 
                                    epsilon_values: List[float]) -> Dict:
        """Evaluate differential privacy effectiveness"""
        results = {}
        dp = DifferentialPrivacy()
        
        for epsilon in epsilon_values:
            dp.epsilon = epsilon
            privacy_loss = []
            utility_loss = []
            
            for update in clean_updates:
                # Add noise
                noisy_update = dp.add_gaussian_noise(update)
                
                # Calculate privacy loss (simplified metric)
                noise_level = torch.norm(noisy_update - update).item()
                privacy_loss.append(1.0 / (1.0 + noise_level))  # Higher noise = better privacy
                
                # Calculate utility loss
                relative_error = noise_level / torch.norm(update).item()
                utility_loss.append(relative_error)
            
            results[epsilon] = {
                'avg_privacy_protection': np.mean(privacy_loss),
                'avg_utility_loss': np.mean(utility_loss),
                'privacy_utility_ratio': np.mean(privacy_loss) / (np.mean(utility_loss) + 1e-8)
            }
        
        return results
    
    def evaluate_byzantine_robustness(self, clean_updates: List[torch.Tensor],
                                    malicious_updates: List[torch.Tensor]) -> Dict:
        """Evaluate Byzantine-robust aggregation methods"""
        aggregator = ByzantineRobustAggregation()
        
        # Combine clean and malicious updates
        all_updates = clean_updates + malicious_updates
        
        # Test different aggregation methods
        results = {}
        
        # 1. Standard averaging (no defense)
        standard_avg = torch.stack(all_updates).mean(dim=0)
        clean_avg = torch.stack(clean_updates).mean(dim=0)
        
        results['standard_averaging'] = {
            'method': 'Standard Averaging',
            'corruption_level': torch.norm(standard_avg - clean_avg).item()
        }
        
        # 2. Krum aggregation
        try:
            num_malicious = len(malicious_updates)
            krum_result = aggregator.krum_aggregation(all_updates, num_malicious)
            results['krum'] = {
                'method': 'Krum',
                'corruption_level': torch.norm(krum_result - clean_avg).item()
            }
        except Exception as e:
            results['krum'] = {'method': 'Krum', 'error': str(e)}
        
        # 3. Trimmed mean
        trimmed_result = aggregator.trimmed_mean_aggregation(all_updates, trim_ratio=0.2)
        results['trimmed_mean'] = {
            'method': 'Trimmed Mean',
            'corruption_level': torch.norm(trimmed_result - clean_avg).item()
        }
        
        return results
    
    def evaluate_intrusion_detection(self, normal_updates: List[torch.Tensor],
                                   malicious_updates: List[torch.Tensor]) -> Dict:
        """Evaluate intrusion detection effectiveness"""
        detector = IntrusionDetector(threshold=2.0)
        
        # Update statistics with normal updates
        detector.update_statistics(normal_updates)
        
        # Test detection rates
        true_positives = 0  # Correctly identified malicious
        false_positives = 0  # Normal updates flagged as malicious
        true_negatives = 0   # Correctly identified normal
        false_negatives = 0  # Malicious updates not detected
        
        # Test normal updates
        for i, update in enumerate(normal_updates):
            is_detected = detector.detect_anomalies(f"normal_client_{i}", update)
            if is_detected:
                false_positives += 1
            else:
                true_negatives += 1
        
        # Test malicious updates
        for i, update in enumerate(malicious_updates):
            is_detected = detector.detect_anomalies(f"malicious_client_{i}", update)
            if is_detected:
                true_positives += 1
            else:
                false_negatives += 1
        
        # Calculate metrics
        total_normal = len(normal_updates)
        total_malicious = len(malicious_updates)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / (total_normal + total_malicious)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }

def run_comprehensive_security_evaluation():
    """Run comprehensive evaluation of security mechanisms"""
    print("=== Comprehensive Security Evaluation ===")
    
    # Create sample data
    num_normal_clients = 10
    num_malicious_clients = 3
    param_size = 1000
    
    # Generate normal client updates (small, similar updates)
    normal_updates = []
    for i in range(num_normal_clients):
        update = torch.randn(param_size) * 0.1  # Small updates
        normal_updates.append(update)
    
    print(f"Generated {len(normal_updates)} normal client updates")
    
    # Generate malicious updates using different attack types
    attack_types = ["random_noise", "sign_flipping", "scaling_attack", "targeted_poisoning"]
    malicious_updates = []
    
    for attack_type in attack_types:
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=1,
            attack_intensity=5.0
        )
        
        attacker = AttackSimulator(attack_config)
        base_update = torch.randn(param_size) * 0.1
        malicious_update = attacker.model_poisoning_attack(base_update)
        malicious_updates.append(malicious_update)
    
    print(f"Generated {len(malicious_updates)} malicious client updates")
    
    # Initialize evaluator
    evaluator = DefenseEvaluator()
    
    # 1. Evaluate Differential Privacy
    print("\n1. Evaluating Differential Privacy...")
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    dp_results = evaluator.evaluate_differential_privacy(normal_updates[:5], epsilon_values)
    
    print("Differential Privacy Results:")
    for epsilon, metrics in dp_results.items():
        print(f"  ε={epsilon}: Privacy={metrics['avg_privacy_protection']:.3f}, "
              f"Utility Loss={metrics['avg_utility_loss']:.3f}, "
              f"Ratio={metrics['privacy_utility_ratio']:.3f}")
    
    # 2. Evaluate Byzantine Robustness
    print("\n2. Evaluating Byzantine Robustness...")
    byzantine_results = evaluator.evaluate_byzantine_robustness(
        normal_updates[:7], malicious_updates[:3]
    )
    
    print("Byzantine Robustness Results:")
    for method, metrics in byzantine_results.items():
        if 'error' in metrics:
            print(f"  {metrics['method']}: Error - {metrics['error']}")
        else:
            print(f"  {metrics['method']}: Corruption Level = {metrics['corruption_level']:.4f}")
    
    # 3. Evaluate Intrusion Detection
    print("\n3. Evaluating Intrusion Detection...")
    intrusion_results = evaluator.evaluate_intrusion_detection(
        normal_updates, malicious_updates
    )
    
    print("Intrusion Detection Results:")
    print(f"  Precision: {intrusion_results['precision']:.3f}")
    print(f"  Recall: {intrusion_results['recall']:.3f}")
    print(f"  F1-Score: {intrusion_results['f1_score']:.3f}")
    print(f"  Accuracy: {intrusion_results['accuracy']:.3f}")
    print(f"  True Positives: {intrusion_results['true_positives']}")
    print(f"  False Positives: {intrusion_results['false_positives']}")
    
    # 4. Gradient Leakage Analysis
    print("\n4. Analyzing Gradient Leakage...")
    attacker = AttackSimulator(AttackConfig("gradient_leakage", 0, 0))
    
    leakage_results = []
    for i, update in enumerate(normal_updates[:3]):
        result = attacker.gradient_leakage_simulation(update)
        leakage_results.append(result)
        print(f"  Client {i}: Leakage Score = {result['leakage_score']:.3f}, "
              f"Risk = {result['privacy_risk']}")
    
    # 5. Generate Visualization
    try:
        create_security_visualization(dp_results, byzantine_results, intrusion_results)
        print("\n📊 Security visualization saved as 'security_evaluation.png'")
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")
    
    print("\n=== Security Evaluation Complete ===")
    
    # Summary and Recommendations
    print("\n📋 SECURITY RECOMMENDATIONS:")
    print("1. Use ε ≤ 1.0 for strong differential privacy")
    print("2. Implement Krum or Trimmed Mean for Byzantine tolerance")
    print("3. Set intrusion detection threshold based on false positive tolerance")
    print("4. Monitor gradient statistics for privacy leakage")
    print("5. Combine multiple defense mechanisms for defense-in-depth")

def create_security_visualization(dp_results, byzantine_results, intrusion_results):
    """Create visualization of security evaluation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Differential Privacy Tradeoff
    epsilons = list(dp_results.keys())
    privacy_scores = [dp_results[eps]['avg_privacy_protection'] for eps in epsilons]
    utility_losses = [dp_results[eps]['avg_utility_loss'] for eps in epsilons]
    
    ax1.plot(epsilons, privacy_scores, 'b-o', label='Privacy Protection', linewidth=2)
    ax1.plot(epsilons, utility_losses, 'r-s', label='Utility Loss', linewidth=2)
    ax1.set_xlabel('Privacy Budget (ε)')
    ax1.set_ylabel('Score')
    ax1.set_title('Differential Privacy: Privacy vs Utility Tradeoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Byzantine Robustness Comparison
    methods = []
    corruption_levels = []
    
    for method, metrics in byzantine_results.items():
        if 'corruption_level' in metrics:
            methods.append(metrics['method'])
            corruption_levels.append(metrics['corruption_level'])
    
    colors = ['red', 'orange', 'green'][:len(methods)]
    bars = ax2.bar(methods, corruption_levels, color=colors, alpha=0.7)
    ax2.set_ylabel('Corruption Level')
    ax2.set_title('Byzantine Robustness: Aggregation Methods Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, corruption_levels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Intrusion Detection Metrics
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [
        intrusion_results['precision'],
        intrusion_results['recall'],
        intrusion_results['f1_score'],
        intrusion_results['accuracy']
    ]
    
    bars = ax3.bar(metrics, values, color=['skyblue', 'lightgreen', 'gold', 'coral'], alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Intrusion Detection Performance')
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Confusion Matrix for Intrusion Detection
    tp = intrusion_results['true_positives']
    fp = intrusion_results['false_positives']
    tn = intrusion_results['true_negatives']
    fn = intrusion_results['false_negatives']
    
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    im = ax4.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, confusion_matrix[i, j], ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
    
    ax4.set_title('Intrusion Detection Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Normal', 'Malicious'])
    ax4.set_yticklabels(['Normal', 'Malicious'])
    
    plt.tight_layout()
    plt.savefig('security_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_comprehensive_security_evaluation()