# -*- coding: utf-8 -*-
"""Conditional Random Field (CRF) layer for sequence labeling.

Based on: https://github.com/kmkurn/pytorch-crf
"""

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field for sequence labeling.

    Computes the log likelihood of tag sequences given emission scores,
    and finds the best tag sequence via Viterbi decoding.

    Args:
        num_tags: Number of tags.
    """

    def __init__(self, num_tags: int) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.04, 0.04)
        nn.init.uniform_(self.end_transitions, -0.04, 0.04)
        nn.init.uniform_(self.transitions, -0.04, 0.04)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Compute the negative log likelihood of the given tag sequence.

        Args:
            emissions: Emission score tensor ``(seq_length, batch_size, num_tags)``.
            tags: Tag tensor ``(seq_length, batch_size)``.
            mask: Mask tensor ``(seq_length, batch_size)``.
            reduce: Whether to average the NLL over the batch.

        Returns:
            NLL scalar (if reduce) or per-sample NLL ``(batch_size,)``.
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        gold_score = self._compute_joint_llh(emissions, tags, mask)
        forward_score = self._compute_log_partition_function(emissions, mask)
        nll = forward_score - gold_score
        return nll if not reduce else torch.mean(nll)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
    ) -> List[List[int]]:
        """Find the most likely tag sequence using the Viterbi algorithm.

        Args:
            emissions: Emission score tensor ``(seq_length, batch_size, num_tags)``.
            mask: Mask tensor ``(seq_length, batch_size)``.

        Returns:
            List of best tag sequences for each sample in the batch.
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        return self._viterbi_decode(emissions, mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_joint_llh(
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        seq_length = emissions.size(0)
        mask = mask.float()

        llh = self.start_transitions[tags[0]]

        for i in range(seq_length - 1):
            cur_tag, next_tag = tags[i], tags[i + 1]
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]
            transition_score = self.transitions[cur_tag, next_tag]
            llh += transition_score * mask[i + 1]

        last_tag_indices = mask.long().sum(0) - 1
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)

        llh += self.end_transitions[last_tags]
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]

        return llh

    def _compute_log_partition_function(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        seq_length = emissions.size(0)
        mask = mask.float()

        log_prob = self.start_transitions.view(1, -1) + emissions[0]

        for i in range(1, seq_length):
            broadcast_log_prob = log_prob.unsqueeze(2)
            broadcast_transitions = self.transitions.unsqueeze(0)
            broadcast_emissions = emissions[i].unsqueeze(1)
            score = broadcast_log_prob + broadcast_transitions + broadcast_emissions
            score = torch.logsumexp(score, 1)
            log_prob = score * mask[i].unsqueeze(1) + log_prob * (1.0 - mask[i]).unsqueeze(1)

        log_prob += self.end_transitions.view(1, -1)
        return torch.logsumexp(log_prob, 1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> List[List[int]]:
        seq_length = emissions.size(0)
        batch_size = emissions.size(1)
        sequence_lengths = mask.long().sum(dim=0)

        best_tags_list = []

        viterbi_score = [self.start_transitions + emissions[0]]
        viterbi_path = []

        for i in range(1, seq_length):
            broadcast_score = viterbi_score[i - 1].view(batch_size, -1, 1)
            broadcast_emission = emissions[i].view(batch_size, 1, -1)
            score = broadcast_score + self.transitions + broadcast_emission
            best_score, best_path = score.max(1)
            viterbi_score.append(best_score)
            viterbi_path.append(best_path)

        for idx in range(batch_size):
            seq_end = sequence_lengths[idx] - 1
            _, best_last_tag = (viterbi_score[seq_end][idx] + self.end_transitions).max(0)
            best_tags = [best_last_tag.item()]

            for path in reversed(viterbi_path[: sequence_lengths[idx] - 1]):
                best_last_tag = path[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
