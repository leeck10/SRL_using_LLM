#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: Train a BERT-CRF model for SRL.

The trained model serves as the encoder for dense example retrieval.

Improvements over previous version:
  - Supports a separate dev set (``crf_training.dev_file``) for early stopping.
  - Falls back to test_file if dev_file is not provided.
  - Uses centralized GPU setup utility.

Usage:
    python scripts/01_train_crf.py --config configs/en_config.yaml
    python scripts/01_train_crf.py --config configs/en_config.yaml --gpu 1
"""

import os
import sys
import json
import random
import argparse

# --- Set GPU BEFORE importing any CUDA libraries ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gpu import set_gpu_before_import, resolve_gpu_id
set_gpu_before_import()

import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, CONFIG_NAME, BertTokenizer
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from models.bert_crf import BertFeatLSTMCRF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class InputExample:
    def __init__(self, sent, verb, feat1, label):
        self.sent = sent
        self.verb = verb
        self.feat1 = feat1
        self.label = label


class InputFeatures:
    def __init__(self, tokens, input_ids, input_mask, segment_ids, label_ids, feat1=None):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.feat1 = feat1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_srl_data_bio(data_path, tokenizer):
    """Load SRL data in CRF format (index word distance label)."""
    temp = []
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if len(line) < 1:
                if not temp:
                    continue
                words = []
                features = []
                verb = tokenizer.tokenize(temp[0].split()[-1])
                labels = []
                for line2 in temp[1:]:
                    parts = line2.split()
                    if len(parts) != 4:
                        continue
                    tokens = tokenizer.tokenize(parts[1])
                    for ii, tok in enumerate(tokens):
                        words.append(tok)
                        features.append(parts[2])  # distance feature
                        if parts[3] != "O":
                            labels.append(("B-" if ii == 0 else "I-") + parts[3])
                        else:
                            labels.append("O")
                data.append(InputExample(sent=words, verb=verb, feat1=features, label=labels))
                temp = []
            else:
                temp.append(line)
    return data


def convert_examples_to_features(examples, seq_length, tokenizer, label_dic, feat_dic):
    """Convert InputExamples to InputFeatures with BERT tokenization."""
    features = []
    for example in examples:
        tokens_a = example.sent + ["[SEP]"] + example.verb

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[: seq_length - 2]

        tokens = ["[CLS]"]
        segment_ids = [0]
        label_ids = [0]
        feat1 = [0]
        seperate = False

        for i, token in enumerate(tokens_a):
            tokens.append(token)
            if token == "[SEP]":
                seperate = True
            segment_ids.append(1 if seperate else 0)
            try:
                label_ids.append(label_dic.get(example.label[i], 0))
            except IndexError:
                label_ids.append(0)
            try:
                feat_val = example.feat1[i]
                feat1.append(feat_dic.get(feat_val, feat_dic.get("[UNK]", 1)))
            except (IndexError, TypeError):
                feat1.append(0)

        tokens.append("[SEP]")
        segment_ids.append(1)
        label_ids.append(0)
        feat1.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Pad to seq_length
        pad_len = seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        segment_ids += [1] * pad_len
        label_ids += [0] * pad_len
        feat1 += [0] * pad_len

        features.append(InputFeatures(
            tokens=tokens, input_ids=input_ids, input_mask=input_mask,
            segment_ids=segment_ids, label_ids=label_ids, feat1=feat1,
        ))
    return features


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_entities(seq):
    """Extract BIO entities as (type, start, end) tuples."""
    prev_tag, prev_type, begin = "O", "", 0
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        tag = chunk[0]
        type_ = chunk.split("-")[-1] if "-" in chunk else chunk
        if prev_tag in ("B", "I") and (tag in ("B", "O") or type_ != prev_type):
            chunks.append((prev_type, begin, i - 1))
        if tag == "B":
            begin = i
        prev_tag, prev_type = tag, type_
    return chunks


def f1_score(y_true, y_pred):
    """Compute micro F1 score for BIO sequences."""
    true_entities = set()
    pred_entities = set()
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        for e in get_entities(yt):
            true_entities.add((i, e))
        for e in get_entities(yp):
            pred_entities.add((i, e))
    correct = len(true_entities & pred_entities)
    p = correct / len(pred_entities) if pred_entities else 0
    r = correct / len(true_entities) if true_entities else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


# ---------------------------------------------------------------------------
# Training / evaluation iteration
# ---------------------------------------------------------------------------

def iteration(model, data_loader, device, epoch, optimizer=None, scheduler=None,
              tokenizer=None, label_list=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    avg_loss, total_correct, total = 0, 0, 0
    y_pred_list, y_true_list = [], []

    desc = f"Epoch:{epoch}" if train else f"Eval:{epoch}"
    data_iter = tqdm(enumerate(data_loader), desc=desc, total=len(data_loader),
                     bar_format="{l_bar}{r_bar}")

    for step, batch in data_iter:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, feat1 = batch

        if train:
            loss, logits, _ = model(input_ids, segment_ids=segment_ids,
                                     input_mask=input_mask, label_ids=label_ids, feat1=feat1)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            result = logits.argmax(dim=-1).cpu()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            avg_loss += loss.item()
        else:
            result = model(input_ids, segment_ids=segment_ids,
                           input_mask=input_mask, label_ids=None, feat1=feat1)

        answer = label_ids[:, 1:]
        mask = answer.ne(0).cpu()
        correct = (result.eq(answer.cpu()) * mask).sum().item()
        total_correct += correct
        total += mask.sum().item()

        if not train and label_list is not None:
            for i in range(result.size(0)):
                y_pred, y_true = [], []
                for j in range(result.size(1)):
                    aid = answer[i, j].item()
                    if aid == 0:
                        break
                    y_pred.append(label_list[result[i, j].item()])
                    y_true.append(label_list[aid])
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

    if not train:
        f1 = f1_score(y_true_list, y_pred_list) * 100
        acc = total_correct / total * 100 if total > 0 else 0
        print(f"  Eval epoch {epoch}: acc={acc:.2f}, F1={f1:.2f}")
        return f1, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BERT-CRF for SRL")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index (overrides config gpu_id)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # GPU was already configured by set_gpu_before_import() above.
    # After CUDA_VISIBLE_DEVICES is set, the only visible GPU is cuda:0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    crf_cfg = cfg["crf_training"]

    random.seed(crf_cfg.get("seed", 42))
    torch.manual_seed(crf_cfg.get("seed", 42))

    output_dir = crf_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(crf_cfg["bert_model"], do_basic_tokenize=True)

    # Load data — train + dev (+ optional test)
    train_examples = load_srl_data_bio(crf_cfg["train_file"], tokenizer)

    # Dev set: use dev_file if provided, otherwise fall back to test_file
    dev_file = crf_cfg.get("dev_file", None)
    if dev_file and os.path.exists(dev_file):
        dev_examples = load_srl_data_bio(dev_file, tokenizer)
        print(f"Train: {len(train_examples)}, Dev: {len(dev_examples)}")
    else:
        dev_examples = load_srl_data_bio(crf_cfg["test_file"], tokenizer)
        print(f"Train: {len(train_examples)}, Dev (from test_file): {len(dev_examples)}")

    # Optional: load separate test set for final evaluation
    test_file = crf_cfg.get("test_file", None)
    test_examples = None
    if test_file and os.path.exists(test_file) and dev_file:
        test_examples = load_srl_data_bio(test_file, tokenizer)
        print(f"Test: {len(test_examples)}")

    # Collect all examples for building vocab (train + dev + test)
    all_examples = list(train_examples) + list(dev_examples)
    if test_examples:
        all_examples += test_examples

    # Build label vocab
    label_dic = {"[PAD]": 0, "O": 1}
    label_list = ["[PAD]", "O"]
    for ex in all_examples:
        for la in ex.label:
            if la not in label_dic:
                label_dic[la] = len(label_dic)
                label_list.append(la)
    with open(os.path.join(output_dir, "label_vocab.txt"), "w") as f:
        for lab in label_list:
            f.write(lab + "\n")

    # Build feature vocab
    feat_dic = {"[PAD]": 0, "[UNK]": 1}
    feat_list = ["[PAD]", "[UNK]"]
    for ex in all_examples:
        if ex.feat1:
            for ft in ex.feat1:
                if ft not in feat_dic:
                    feat_dic[ft] = len(feat_dic)
                    feat_list.append(ft)
    with open(os.path.join(output_dir, "feat_vocab.txt"), "w") as f:
        for ft in feat_list:
            f.write(ft + "\n")

    # Convert to features
    max_len = crf_cfg.get("max_sent_length", 156)
    train_features = convert_examples_to_features(train_examples, max_len, tokenizer, label_dic, feat_dic)
    dev_features = convert_examples_to_features(dev_examples, max_len, tokenizer, label_dic, feat_dic)

    # Build DataLoaders
    def make_dataset(features):
        return TensorDataset(
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.input_mask for f in features], dtype=torch.long),
            torch.tensor([f.segment_ids for f in features], dtype=torch.long),
            torch.tensor([f.label_ids for f in features], dtype=torch.long),
            torch.tensor([f.feat1 for f in features], dtype=torch.long),
        )

    batch_size = crf_cfg.get("batch_size", 24)
    train_ds = make_dataset(train_features)
    dev_ds = make_dataset(dev_features)
    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size)
    dev_loader = DataLoader(dev_ds, sampler=SequentialSampler(dev_ds), batch_size=batch_size)

    # Build model
    model = BertFeatLSTMCRF.from_pretrained(
        crf_cfg["bert_model"], num_labels=len(label_dic),
        feat_vocab_size=len(feat_dic), feat_embed_dim=crf_cfg.get("feat_embed_dim", 100),
        feat_num=1, rnn_layers=crf_cfg.get("rnn_layers", 1),
        dropout_rnn=crf_cfg.get("dropout_rnn", 0.1),
        rnn_type=crf_cfg.get("rnn_type", "lstm"),
        rnn_hidden_size=crf_cfg.get("rnn_hidden_size", 0),
    )
    model.to(device)

    # Optimizer & Scheduler
    num_epochs = crf_cfg.get("num_train_epochs", 30)
    num_steps = (len(train_examples) // batch_size + 1) * num_epochs
    optimizer = AdamW(
        model.parameters(),
        lr=crf_cfg.get("learning_rate", 2e-5),
    )
    warmup_steps = int(num_steps * crf_cfg.get("warmup_proportion", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)

    # Training loop with early stopping on dev set
    best_f1 = -1
    best_epoch = -1
    for epoch in trange(num_epochs, desc="Epochs"):
        iteration(model, train_loader, device, epoch, optimizer, scheduler, train=True)

        with torch.no_grad():
            model.eval()
            f1, acc = iteration(model, dev_loader, device, epoch,
                                tokenizer=tokenizer, label_list=label_list, train=False)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
            with open(os.path.join(output_dir, CONFIG_NAME), "w") as f:
                f.write(model_to_save.config.to_json_string())

    print(f"\nBest epoch: {best_epoch}, Best Dev F1: {best_f1:.2f}")
    print(f"Model saved to: {output_dir}")

    # Optional: final evaluation on test set
    if test_examples:
        print("\nRunning final evaluation on test set...")
        test_features_final = convert_examples_to_features(
            test_examples, max_len, tokenizer, label_dic, feat_dic
        )
        test_ds = make_dataset(test_features_final)
        test_loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=batch_size)

        # Load best model
        best_state = torch.load(os.path.join(output_dir, WEIGHTS_NAME), map_location=device)
        model_to_eval = model.module if hasattr(model, "module") else model
        model_to_eval.load_state_dict(best_state)

        with torch.no_grad():
            model.eval()
            test_f1, test_acc = iteration(
                model, test_loader, device, best_epoch,
                tokenizer=tokenizer, label_list=label_list, train=False,
            )
        print(f"Test F1: {test_f1:.2f}, Test Acc: {test_acc:.2f}")


if __name__ == "__main__":
    main()
