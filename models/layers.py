# -*- coding: utf-8 -*-
"""Custom RNN layers for sequence encoding.

Based on: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedBRNN(nn.Module):
    """Stacked Bidirectional RNN (LSTM/GRU).

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension per direction.
        num_layers: Number of stacked RNN layers.
        dropout_rate: Dropout probability between layers.
        dropout_output: Whether to apply dropout on the final output.
        rnn_type: RNN cell type (default: ``nn.LSTM``).
        concat_layers: If True, concatenate all layer outputs.
        padding: Whether to use packed sequences (slower but precise).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        dropout_output: bool = False,
        rnn_type=nn.LSTM,
        concat_layers: bool = False,
        padding: bool = False,
    ):
        super().__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(
                rnn_type(in_size, hidden_size, num_layers=1, bidirectional=True)
            )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(batch, seq_len, input_size)``.
            x_mask: Padding mask ``(batch, seq_len)`` where 1 = pad.

        Returns:
            Encoded output ``(batch, seq_len, hidden_size * 2)``.
        """
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            self.rnns[i].flatten_parameters()
            rnn_output, _ = self.rnns[i](rnn_input)
            outputs.append(rnn_output)

        output = torch.cat(outputs[1:], 2) if self.concat_layers else outputs[-1]
        output = output.transpose(0, 1)

        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output

    def _forward_padded(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)

        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=True)

        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                dropout_input = F.dropout(
                    rnn_input.data, p=self.dropout_rate, training=self.training
                )
                rnn_input = nn.utils.rnn.PackedSequence(
                    dropout_input, rnn_input.batch_sizes
                )
            outputs.append(self.rnns[i](rnn_input)[0])

        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        output = torch.cat(outputs[1:], 2) if self.concat_layers else outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output
