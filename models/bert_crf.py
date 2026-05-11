# -*- coding: utf-8 -*-
"""BERT + Feature Embedding + BiLSTM + CRF model for SRL.

Architecture:
    BERT encoder -> Dropout -> [Feature Embedding] -> BiLSTM -> Linear -> CRF
"""

import sys

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel

from models.crf import CRF
from models.layers import StackedBRNN


class BertFeatLSTMCRF(BertPreTrainedModel):
    """BERT with feature embedding, BiLSTM, and CRF for token classification.

    Args:
        config: BERT config.
        num_labels: Number of output labels.
        feat_vocab_size: Size of the feature vocabulary.
        feat_embed_dim: Dimension of feature embeddings.
        feat_num: Number of feature inputs (1 or 2).
        rnn_layers: Number of BiLSTM layers.
        dropout_rnn: Dropout rate for BiLSTM.
        rnn_type: RNN type string (``'lstm'`` or ``'gru'``).
        rnn_hidden_size: BiLSTM hidden size (0 = use BERT hidden size).
    """

    def __init__(
        self,
        config,
        num_labels: int = None,
        feat_vocab_size: int = 100,
        feat_embed_dim: int = 100,
        feat_num: int = 1,
        rnn_layers: int = 1,
        dropout_rnn: float = 0.1,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 0,
    ):
        super().__init__(config)
        if num_labels is None:
            num_labels = getattr(config, "num_labels", None)

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(rnn_type, nn.LSTM)
        if rnn_hidden_size <= 0:
            rnn_hidden_size = config.hidden_size

        print(f"Feat embedding dim: {feat_embed_dim}", file=sys.stderr)
        print(f"RNN layers: {rnn_layers}", file=sys.stderr)
        print(f"RNN hidden size: {rnn_hidden_size}", file=sys.stderr)

        # BERT encoder
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Feature embedding
        self.embed = nn.Embedding(feat_vocab_size, feat_embed_dim)
        self.feat_num = feat_num

        # BiLSTM
        self.rnn = StackedBRNN(
            input_size=config.hidden_size + feat_num * feat_embed_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout_rate=dropout_rnn,
            dropout_output=True,
            rnn_type=rnn_cls,
            padding=False,
        )

        # Classifier + CRF
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor = None,
        segment_ids: torch.Tensor = None,
        label_ids: torch.Tensor = None,
        feat1: torch.Tensor = None,
        feat2: torch.Tensor = None,
    ):
        """Forward pass.

        Args:
            input_ids: Token IDs ``(batch, seq_len)``.
            input_mask: Attention mask ``(batch, seq_len)``.
            segment_ids: Segment IDs ``(batch, seq_len)``.
            label_ids: Gold label IDs ``(batch, seq_len)`` or None for inference.
            feat1: Feature 1 IDs ``(batch, seq_len)``.
            feat2: Feature 2 IDs ``(batch, seq_len)``.

        Returns:
            If label_ids is given: ``(loss, logits, bert_output)``
            Otherwise: predicted tag IDs ``(batch, seq_len-1)``
        """
        outputs = self.bert(input_ids, input_mask, segment_ids)
        sequence_output = self.dropout(outputs["last_hidden_state"])

        # Concatenate feature embeddings
        if self.feat_num >= 1 and feat1 is not None:
            feat1_emb = self.dropout(
                self.embed(feat1).view(feat1.size(0), feat1.size(1), -1)
            )
            x = torch.cat((sequence_output, feat1_emb), dim=-1)
        else:
            x = sequence_output

        if self.feat_num >= 2 and feat2 is not None:
            feat2_emb = self.dropout(
                self.embed(feat2).view(feat2.size(0), feat2.size(1), -1)
            )
            x = torch.cat((x, feat2_emb), dim=-1)

        # BiLSTM
        rnn_mask = input_ids.eq(0)
        h = self.rnn._forward_unpadded(x, rnn_mask)

        # Classifier
        logits = self.classifier(h)

        # CRF — transpose to (seq_len, batch, labels)
        logits = logits.transpose(0, 1)

        if label_ids is not None:
            label_ids = label_ids.transpose(0, 1)
            mask = label_ids.eq(0).eq(0)
            # Skip [CLS] token (index 0)
            loss = self.crf(logits[1:], label_ids[1:], mask[1:])
            return loss, logits[1:].transpose(0, 1), sequence_output
        else:
            result = self.crf.decode(logits[1:])
            return torch.LongTensor(result)
