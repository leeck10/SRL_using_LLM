# -*- coding: utf-8 -*-
"""PyTorch tensor-based similarity search database.

Supports Euclidean and Mahalanobis distance metrics for dense vector retrieval.
"""

from typing import List, Tuple

import torch
import pickle


class VectorDatabase:
    """Dense vector database with GPU-accelerated distance computation.

    Stores vectors as PyTorch tensors and computes distances on the fly.

    Args:
        metric: Distance metric — ``'euclidean'`` or ``'mahalanobis'``.
        use_gpu: Whether to keep vectors on GPU.
    """

    def __init__(self, metric: str = "euclidean", use_gpu: bool = True):
        if metric not in ("euclidean", "mahalanobis"):
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'mahalanobis'.")
        self.metric = metric
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.data: List[Tuple[torch.Tensor, int]] = []
        self.inv_covariance_matrix = None

    def add_item(self, vector: torch.Tensor, index: int):
        """Add a vector with its associated index.

        Args:
            vector: 1-D tensor.
            index: Integer index (typically the position in the training set).
        """
        self.data.append((vector, index))

    def update_covariance_matrix(self):
        """Compute the inverse covariance matrix for Mahalanobis distance.

        Must be called after all items are added and before searching
        with ``metric='mahalanobis'``.
        """
        if self.metric != "mahalanobis":
            return
        all_vectors = torch.stack([vec for vec, _ in self.data])
        cov_matrix = torch.cov(all_vectors.T)
        self.inv_covariance_matrix = torch.linalg.pinv(cov_matrix)

    def _calculate_distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute distance between two vectors.

        Args:
            x: Query vector.
            y: Database vector.

        Returns:
            Scalar distance value.
        """
        if self.metric == "euclidean":
            return torch.linalg.norm(x - y).item()
        else:  # mahalanobis
            diff = x - y
            return torch.sqrt(diff @ self.inv_covariance_matrix @ diff).item()

    def search(self, query_vector: torch.Tensor, k: int = 1) -> List[Tuple[float, int]]:
        """Find the k nearest neighbors to the query vector.

        Args:
            query_vector: 1-D query tensor.
            k: Number of neighbors to return.

        Returns:
            List of ``(distance, index)`` tuples sorted by distance (ascending).
        """
        distances = []
        for stored_vector, index in self.data:
            distance = self._calculate_distance(query_vector, stored_vector)
            distances.append((distance, index))
        distances.sort(key=lambda x: x[0])
        return distances[:k]

    def save(self, filepath: str):
        """Save the database to a pickle file.

        Args:
            filepath: Output file path (without extension).
        """
        save_data = {
            "data": [(v.cpu(), idx) for v, idx in self.data],
            "inv_covariance_matrix": (
                self.inv_covariance_matrix.cpu()
                if self.inv_covariance_matrix is not None
                else None
            ),
            "metric": self.metric,
        }
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath: str, use_gpu: bool = True) -> "VectorDatabase":
        """Load a database from a pickle file.

        Args:
            filepath: Input file path (without extension).
            use_gpu: Whether to move vectors to GPU.

        Returns:
            Loaded VectorDatabase instance.
        """
        with open(f"{filepath}.pkl", "rb") as f:
            save_data = pickle.load(f)

        db = cls(metric=save_data["metric"], use_gpu=use_gpu)

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        db.data = [(v.to(device), idx) for v, idx in save_data["data"]]
        if save_data["inv_covariance_matrix"] is not None:
            db.inv_covariance_matrix = save_data["inv_covariance_matrix"].to(device)
        return db
