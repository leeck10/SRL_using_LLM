# -*- coding: utf-8 -*-
"""BERT-based sentence encoder for example retrieval.

Supports two modes:
  - ``"crf"``: Load from a fine-tuned BERT-CRF checkpoint (extracts the BERT
    backbone weights from the ``BertFeatLSTMCRF`` state dict).
  - ``"pretrained"``: Load a vanilla pre-trained BERT model directly from
    HuggingFace (e.g., ``bert-base-uncased``, ``klue/bert-base``).

Both modes encode sentences using the [CLS] token from the last hidden layer.
"""

import os

import torch
import numpy as np
from transformers import BertTokenizer, BertModel


class BertEncoder:
    """Encode sentences into dense vectors using BERT [CLS] token.

    Args:
        bert_model_name: HuggingFace model name or local checkpoint path.
        use_gpu: Whether to use GPU acceleration.
        encoder_type: ``"crf"`` to load from a fine-tuned BERT-CRF checkpoint,
            or ``"pretrained"`` to load a vanilla BERT model.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        use_gpu: bool = True,
        encoder_type: str = "crf",
    ):
        self.encoder_type = encoder_type

        if encoder_type == "pretrained":
            # Load vanilla pre-trained BERT directly
            self.tokenizer = BertTokenizer.from_pretrained(
                bert_model_name, do_basic_tokenize=True
            )
            self.bert_model = BertModel.from_pretrained(bert_model_name)
        else:
            # Load from fine-tuned BERT-CRF checkpoint directory
            self.tokenizer = BertTokenizer.from_pretrained(
                bert_model_name, do_basic_tokenize=True
            )
            self.bert_model = BertModel.from_pretrained(bert_model_name)

        self.vector_dim = self.bert_model.config.hidden_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.bert_model = self.bert_model.cuda()
        self.bert_model.eval()

    def encode(self, sentence: str) -> torch.Tensor:
        """Encode a sentence into a vector using the [CLS] token.

        Args:
            sentence: Input text.

        Returns:
            1-D tensor of shape ``(hidden_size,)``.
        """
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_hidden_states=False)

        # Last hidden state, [CLS] token
        vector = outputs.last_hidden_state[:, 0, :].flatten()
        return vector
