# -*- coding: utf-8 -*-
"""Example selection strategies: Top-K and MMR (Maximal Marginal Relevance)."""

from typing import List, Tuple

import torch

from retrieval.database import VectorDatabase


def select_topk(
    db: VectorDatabase, query_vector: torch.Tensor, k: int = 5
) -> List[Tuple[float, int]]:
    """Select the top-k most similar examples by distance.

    Args:
        db: VectorDatabase instance with stored training examples.
        query_vector: Query vector (1-D tensor).
        k: Number of examples to select.

    Returns:
        List of ``(distance, index)`` tuples.
    """
    return db.search(query_vector, k=k)


def select_mmr(
    db: VectorDatabase,
    query_vector: torch.Tensor,
    k: int = 5,
    lambda_param: float = 0.7,
    candidate_pool_size: int = 50,
) -> List[Tuple[float, int]]:
    """Select examples using Maximal Marginal Relevance.

    Balances relevance to the query with diversity among selected examples.

    MMR(d) = lambda * Sim(d, q) - (1 - lambda) * max_{d' in S} Sim(d, d')

    Args:
        db: VectorDatabase instance.
        query_vector: Query vector (1-D tensor).
        k: Number of examples to select.
        lambda_param: Trade-off parameter (higher = more relevance, less diversity).
        candidate_pool_size: Size of the initial candidate pool from Top-K.

    Returns:
        List of ``(distance, index)`` tuples in selection order.
    """
    # Get a larger candidate pool first
    candidates = db.search(query_vector, k=candidate_pool_size)

    # Gather candidate vectors for diversity computation
    index_to_vector = {}
    for vec, idx in db.data:
        index_to_vector[idx] = vec

    selected: List[Tuple[float, int]] = []
    remaining = list(candidates)

    for _ in range(min(k, len(remaining))):
        best_score = float("-inf")
        best_idx = -1
        best_item = None

        for i, (dist, idx) in enumerate(remaining):
            # Relevance: negative distance (smaller distance = higher relevance)
            relevance = -dist

            # Diversity: maximum similarity to already selected items
            max_sim_to_selected = 0.0
            if selected:
                cand_vec = index_to_vector.get(idx)
                if cand_vec is not None:
                    for _, sel_idx in selected:
                        sel_vec = index_to_vector.get(sel_idx)
                        if sel_vec is not None:
                            # Cosine similarity
                            sim = torch.dot(cand_vec, sel_vec) / (
                                torch.linalg.norm(cand_vec) * torch.linalg.norm(sel_vec) + 1e-8
                            )
                            max_sim_to_selected = max(max_sim_to_selected, sim.item())

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
                best_item = (dist, idx)

        if best_item is not None:
            selected.append(best_item)
            remaining.pop(best_idx)

    return selected
