"""
Deep Tabular Autoencoder Detector for Behavioral Fraud.
Compresses multi-dimensional behavioral profiles into a low-dimensional manifold
and uses reconstruction error as the anomaly score, with per-feature error attribution.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from .base import BaseAnomalyDetector


class TabularAEModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 12, hidden_dim: int = 48, dropout_rate: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderDetector(BaseAnomalyDetector):
    def __init__(
        self,
        latent_dim: int = 12,
        hidden_dim: int = 48,
        epochs: int = 25,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        random_state: int = 42,
    ):
        super().__init__(name="Deep Autoencoder (AE)")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.network = None
        self.device = torch.device("cpu")  # CPU is faster for tabular batch sizes < 50k

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "AutoencoderDetector":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X_scaled = self.scaler.fit_transform(X)
        input_dim = X.shape[1]
        
        # Adaptive latent dimension based on input dimension
        actual_latent = min(self.latent_dim, max(2, input_dim // 3))
        actual_hidden = max(actual_latent * 2, min(self.hidden_dim, input_dim * 2))
        
        self.network = TabularAEModule(
            input_dim=input_dim,
            latent_dim=actual_latent,
            hidden_dim=actual_hidden,
        ).to(self.device)
        
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        
        self.network.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.network(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Computes reconstruction error (MSE per sample).
        Higher MSE = less conformant with normal behavioral manifold = higher risk.
        """
        self.network.eval()
        X_scaled = self.scaler.transform(X)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.network(tensor_x)
            # Per-sample mean squared error across all features
            sample_errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1).cpu().numpy()
            
        return sample_errors

    def explain_sample_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Returns per-feature squared error matrix (N x D) for root-cause attribution.
        """
        self.network.eval()
        X_scaled = self.scaler.transform(X)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.network(tensor_x)
            feature_errors = ((tensor_x - reconstructed) ** 2).cpu().numpy()
            
        return feature_errors
