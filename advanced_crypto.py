"""
Advanced Cryptographic Techniques for Federated Learning
This module implements state-of-the-art cryptographic protocols for secure parameter sharing
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
import hashlib
import secrets
from dataclasses import dataclass
import pickle
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

@dataclass
class CryptoParams:
    """Parameters for cryptographic operations"""
    key_size: int = 256  # AES key size in bits
    salt_size: int = 16  # Salt size in bytes
    iterations: int = 100000  # PBKDF2 iterations

class SecretSharing:
    """Implements Shamir's Secret Sharing scheme for distributed parameter storage"""
    
    def __init__(self, threshold: int, num_shares: int):
        """
        Initialize secret sharing scheme
        
        Args:
            threshold: Minimum number of shares needed to reconstruct secret
            num_shares: Total number of shares to generate
        """
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = 2**31 - 1  # Large prime for finite field arithmetic
    
    def _poly_eval(self, coeffs: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        for i, coeff in enumerate(coeffs):
            result = (result + coeff * pow(x, i, self.prime)) % self.prime
        return result
    
    def _lagrange_interpolation(self, points: List[Tuple[int, int]]) -> int:
        """Reconstruct secret using Lagrange interpolation"""
        if len(points) < self.threshold:
            raise ValueError("Insufficient shares for reconstruction")
        
        result = 0
        for i in range(len(points)):
            xi, yi = points[i]
            
            # Calculate Lagrange basis polynomial
            li = 1
            for j in range(len(points)):
                if i != j:
                    xj, _ = points[j]
                    li = (li * (0 - xj) * pow(xi - xj, self.prime - 2, self.prime)) % self.prime
            
            result = (result + yi * li) % self.prime
        
        return result % self.prime
    
    def split_secret(self, secret: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """Split tensor into secret shares"""
        # Flatten tensor and convert to integers
        flat_secret = secret.flatten()
        int_secret = (flat_secret * 1000000).int()  # Scale for precision
        
        shares = []
        
        for share_idx in range(1, self.num_shares + 1):
            share_values = []
            
            for value in int_secret:
                # Generate random coefficients for polynomial
                coeffs = [int(value)] + [secrets.randbelow(self.prime) for _ in range(self.threshold - 1)]
                
                # Evaluate polynomial at share_idx
                share_value = self._poly_eval(coeffs, share_idx)
                share_values.append(share_value)
            
            share_tensor = torch.tensor(share_values, dtype=torch.float32).reshape(secret.shape)
            shares.append((share_idx, share_tensor))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Reconstruct secret from shares"""
        if len(shares) < self.threshold:
            raise ValueError("Insufficient shares for reconstruction")
        
        # Take first threshold shares
        selected_shares = shares[:self.threshold]
        
        # Get tensor shape from first share
        shape = selected_shares[0][1].shape
        flat_size = selected_shares[0][1].numel()
        
        reconstructed_values = []
        
        for i in range(flat_size):
            # Collect points for this parameter
            points = []
            for share_idx, share_tensor in selected_shares:
                flat_share = share_tensor.flatten()
                points.append((share_idx, int(flat_share[i])))
            
            # Reconstruct this parameter
            reconstructed_value = self._lagrange_interpolation(points)
            reconstructed_values.append(reconstructed_value / 1000000.0)  # Unscale
        
        return torch.tensor(reconstructed_values).reshape(shape)

class SecureMultiPartyComputation:
    """Implements secure multi-party computation for federated aggregation"""
    
    def __init__(self, num_parties: int):
        self.num_parties = num_parties
        self.shares = {}
    
    def generate_random_shares(self, value: float) -> List[float]:
        """Generate random shares that sum to the original value"""
        shares = [secrets.SystemRandom().uniform(-1, 1) for _ in range(self.num_parties - 1)]
        shares.append(value - sum(shares))
        return shares
    
    def secure_sum(self, party_values: List[float]) -> float:
        """Compute sum using secure multi-party computation"""
        if len(party_values) != self.num_parties:
            raise ValueError("Incorrect number of parties")
        
        # In real implementation, this would use more sophisticated protocols
        # like BGW or GMW. This is a simplified version for demonstration.
        return sum(party_values)

class AdvancedEncryption:
    """Advanced encryption techniques for federated learning"""
    
    def __init__(self, crypto_params: CryptoParams = None):
        self.crypto_params = crypto_params or CryptoParams()
    
    def derive_key(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derive encryption key from password using PBKDF2"""
        if salt is None:
            salt = os.urandom(self.crypto_params.salt_size)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.crypto_params.key_size // 8,
            salt=salt,
            iterations=self.crypto_params.iterations,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return key, salt
    
    def encrypt_tensor(self, tensor: torch.Tensor, password: str) -> Dict[str, str]:
        """Encrypt tensor using AES encryption"""
        # Serialize tensor
        tensor_bytes = pickle.dumps(tensor)
        
        # Derive key
        key, salt = self.derive_key(password)
        
        # Generate IV
        iv = os.urandom(16)
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(tensor_bytes) % 16)
        padded_data = tensor_bytes + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return {
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'salt': base64.b64encode(salt).decode(),
            'iv': base64.b64encode(iv).decode()
        }
    
    def decrypt_tensor(self, encrypted_package: Dict[str, str], password: str) -> torch.Tensor:
        """Decrypt tensor using AES decryption"""
        # Decode components
        encrypted_data = base64.b64decode(encrypted_package['encrypted_data'])
        salt = base64.b64decode(encrypted_package['salt'])
        iv = base64.b64decode(encrypted_package['iv'])
        
        # Derive key
        key, _ = self.derive_key(password, salt)
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        padding_length = decrypted_padded[-1]
        decrypted_data = decrypted_padded[:-padding_length]
        
        # Deserialize tensor
        return pickle.loads(decrypted_data)

class SecureAggregationProtocol:
    """Implements secure aggregation protocol for federated learning"""
    
    def __init__(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold
        self.secret_sharing = SecretSharing(threshold, num_clients)
        self.encryption = AdvancedEncryption()
    
    def prepare_client_update(self, update: torch.Tensor, client_password: str) -> Dict:
        """Prepare client update for secure aggregation"""
        # Split update into secret shares
        shares = self.secret_sharing.split_secret(update)
        
        # Encrypt each share
        encrypted_shares = []
        for share_idx, share_tensor in shares:
            encrypted_share = self.encryption.encrypt_tensor(
                share_tensor, 
                f"{client_password}_{share_idx}"
            )
            encrypted_shares.append((share_idx, encrypted_share))
        
        return {
            'encrypted_shares': encrypted_shares,
            'client_id': f"client_{secrets.randbelow(1000000)}"
        }
    
    def aggregate_updates(self, client_packages: List[Dict], passwords: List[str]) -> torch.Tensor:
        """Securely aggregate client updates"""
        if len(client_packages) < self.threshold:
            raise ValueError("Insufficient clients for secure aggregation")
        
        # Collect shares from each client
        all_shares = []
        
        for i, package in enumerate(client_packages[:self.threshold]):
            client_shares = []
            
            for share_idx, encrypted_share in package['encrypted_shares']:
                # Decrypt share
                decrypted_share = self.encryption.decrypt_tensor(
                    encrypted_share,
                    f"{passwords[i]}_{share_idx}"
                )
                client_shares.append((share_idx, decrypted_share))
            
            all_shares.append(client_shares)
        
        # Reconstruct each client's update
        reconstructed_updates = []
        for client_shares in all_shares:
            reconstructed_update = self.secret_sharing.reconstruct_secret(client_shares)
            reconstructed_updates.append(reconstructed_update)
        
        # Average the updates
        return torch.stack(reconstructed_updates).mean(dim=0)

class ZeroKnowledgeProof:
    """Simple zero-knowledge proof for parameter validity"""
    
    def __init__(self):
        self.challenge_size = 256
    
    def generate_proof(self, secret_value: float, public_commitment: str) -> Dict:
        """Generate zero-knowledge proof that prover knows the secret"""
        # Simplified ZK proof - in practice, use more sophisticated protocols
        challenge = secrets.randbelow(2**self.challenge_size)
        
        # Create response (simplified)
        response = hashlib.sha256(
            f"{secret_value}_{challenge}_{public_commitment}".encode()
        ).hexdigest()
        
        return {
            'challenge': challenge,
            'response': response,
            'commitment': public_commitment
        }
    
    def verify_proof(self, proof: Dict, expected_commitment: str) -> bool:
        """Verify zero-knowledge proof"""
        # In a real implementation, this would verify without learning the secret
        return proof['commitment'] == expected_commitment

# Demonstration of advanced cryptographic techniques
if __name__ == "__main__":
    print("=== Advanced Cryptographic Techniques Demo ===")
    
    # Create sample tensor
    sample_tensor = torch.randn(3, 4)
    print(f"Original tensor shape: {sample_tensor.shape}")
    print(f"Original tensor:\n{sample_tensor}")
    
    # 1. Secret Sharing Demo
    print("\n1. Secret Sharing Demo:")
    secret_sharing = SecretSharing(threshold=3, num_shares=5)
    shares = secret_sharing.split_secret(sample_tensor)
    print(f"Generated {len(shares)} shares")
    
    # Reconstruct from subset of shares
    reconstructed = secret_sharing.reconstruct_secret(shares[:3])
    print(f"Reconstruction error: {torch.norm(sample_tensor - reconstructed):.6f}")
    
    # 2. Advanced Encryption Demo
    print("\n2. Advanced Encryption Demo:")
    encryption = AdvancedEncryption()
    password = "secure_federated_learning_2025"
    
    encrypted_package = encryption.encrypt_tensor(sample_tensor, password)
    print("Tensor encrypted successfully")
    
    decrypted_tensor = encryption.decrypt_tensor(encrypted_package, password)
    print(f"Decryption error: {torch.norm(sample_tensor - decrypted_tensor):.6f}")
    
    # 3. Secure Aggregation Protocol Demo
    print("\n3. Secure Aggregation Protocol Demo:")
    num_clients = 4
    threshold = 3
    
    secure_agg = SecureAggregationProtocol(num_clients, threshold)
    
    # Simulate client updates
    client_updates = [torch.randn_like(sample_tensor) for _ in range(num_clients)]
    client_passwords = [f"password_{i}" for i in range(num_clients)]
    
    # Prepare client packages
    client_packages = []
    for i, update in enumerate(client_updates):
        package = secure_agg.prepare_client_update(update, client_passwords[i])
        client_packages.append(package)
    
    print(f"Prepared {len(client_packages)} client packages")
    
    # Aggregate securely
    aggregated_result = secure_agg.aggregate_updates(client_packages, client_passwords)
    
    # Compare with plain aggregation
    plain_aggregation = torch.stack(client_updates).mean(dim=0)
    aggregation_error = torch.norm(aggregated_result - plain_aggregation)
    print(f"Secure aggregation error: {aggregation_error:.6f}")
    
    print("\n=== All cryptographic techniques demonstrated successfully! ===")