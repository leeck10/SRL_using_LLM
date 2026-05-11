#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end SRL inference pipeline.

Given a sentence with a known predicate (sense number + position),
performs the full pipeline: encode -> retrieve -> order -> generate -> parse.

IMPORTANT: If using this module as a standalone script, call
``set_gpu_before_import()`` **before** importing this module.
When using ``from_config()``, GPU setup is handled automatically
only if ``CUDA_VISIBLE_DEVICES`` is set before import.

Usage (correct GPU handling)::

    import os, sys
    sys.path.insert(0, "path/to/SRL-ICL3")
    from utils.gpu import set_gpu_before_import
    set_gpu_before_import()   # reads --gpu / --config, sets env var

    from inference.pipeline import SRLPipeline
    pipe = SRLPipeline.from_config("configs/en_config.yaml")
    result = pipe.predict(
        sentence="In an Oct. 19 review of The Misanthrope ...",
        predicate="attribute.01",
        predicate_index=8,
        output_format="dict",
    )
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io import load_json
from retrieval.encoder import BertEncoder
from retrieval.database import VectorDatabase
from retrieval.selector import select_topk, select_mmr
from prompts.builder import build_prompt
from ordering.cone import load_order_file


class SRLPipeline:
    """End-to-end SRL inference pipeline.

    Assumes that the predicate has already been identified
    (predicate identification is a given).

    Args:
        encoder: BertEncoder for query encoding.
        db: VectorDatabase with encoded training examples.
        train_data: List of training example dicts.
        framefiles: Dict mapping verb sense -> role definitions.
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        language: ``'en'`` or ``'ko'``.
        default_order: Default example ordering indices.
        default_format: Default output format (``'conll'`` or ``'dict'``).
        num_examples: Number of in-context examples.
        selection_strategy: ``'topk'`` or ``'mmr'``.
        mmr_lambda: Lambda parameter for MMR selection.
    """

    def __init__(
        self,
        encoder: BertEncoder,
        db: VectorDatabase,
        train_data: List[Dict],
        framefiles: Dict,
        model,
        tokenizer,
        language: str = "en",
        default_order: List[int] = None,
        default_format: str = "dict",
        num_examples: int = 5,
        selection_strategy: str = "topk",
        mmr_lambda: float = 0.7,
    ):
        self.encoder = encoder
        self.db = db
        self.train_data = train_data
        self.framefiles = framefiles
        self.model = model
        self.tokenizer = tokenizer
        self.language = language
        self.default_order = default_order or list(range(num_examples))
        self.default_format = default_format
        self.num_examples = num_examples
        self.selection_strategy = selection_strategy
        self.mmr_lambda = mmr_lambda

    @classmethod
    def from_config(cls, config_path: str, gpu_id: int = None) -> "SRLPipeline":
        """Create a pipeline from a YAML config file.

        Args:
            config_path: Path to the config YAML file.
            gpu_id: CUDA device index. Overrides config ``gpu_id`` if given.

        Returns:
            Initialized SRLPipeline.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # GPU must be configured BEFORE this module is imported.
        # See module docstring for correct usage.
        # After set_gpu_before_import(), the only visible GPU is cuda:0.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ret_cfg = cfg["retrieval"]
        data_cfg = cfg["data"]
        cone_cfg = cfg.get("cone", {})

        # Determine which LLM config to use (eval_llm preferred, fallback to llm)
        llm_cfg = cfg.get("eval_llm", cfg.get("llm", {}))

        # Determine encoder type
        encoder_type = ret_cfg.get("encoder_type", "crf")
        if encoder_type == "pretrained":
            encoder_model = ret_cfg.get("pretrained_model", "bert-base-uncased")
        else:
            encoder_model = ret_cfg["encoder_model"]

        print("Loading encoder...")
        encoder = BertEncoder(
            bert_model_name=encoder_model, use_gpu=True, encoder_type=encoder_type
        )

        print("Loading retrieval database...")
        db = VectorDatabase.load(ret_cfg["db_path"], use_gpu=True)

        print("Loading training data...")
        train_json_path = ret_cfg["db_path"] + "_train_data.json"
        with open(train_json_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        print("Loading framefiles...")
        framefiles = load_json(data_cfg["framefiles"])

        print(f"Loading LLM: {llm_cfg['model_id']}...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_id"])
        model = AutoModelForCausalLM.from_pretrained(
            llm_cfg["model_id"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        # Load ordering
        order_file = cone_cfg.get("order_output", None)
        if order_file and os.path.exists(order_file):
            default_order = load_order_file(order_file)
            print(f"Loaded ordering from {order_file}: {default_order}")
        else:
            num_ex = cone_cfg.get("num_examples", 5)
            default_order = list(range(num_ex))
            print(f"Using default ordering: {default_order}")

        return cls(
            encoder=encoder,
            db=db,
            train_data=train_data,
            framefiles=framefiles,
            model=model,
            tokenizer=tokenizer,
            language=cfg["language"],
            default_order=default_order,
            default_format=cfg.get("output_format", "dict"),
            num_examples=cone_cfg.get("num_examples", 5),
            selection_strategy=ret_cfg.get("strategy", "topk"),
            mmr_lambda=ret_cfg.get("mmr_lambda", 0.7),
        )

    def predict(
        self,
        sentence: str,
        predicate: str,
        predicate_index: int,
        output_format: Optional[str] = None,
        num_examples: Optional[int] = None,
        example_order: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> Dict:
        """Run SRL inference on a single sentence.

        Args:
            sentence: Input sentence (plain text, no tags).
            predicate: Predicate sense ID (e.g., ``'attribute.01'``).
            predicate_index: 0-based word/eojeol index of the predicate.
            output_format: ``'conll'`` or ``'dict'`` (default: pipeline default).
            num_examples: Number of examples to retrieve (default: pipeline default).
            example_order: Example ordering (default: pipeline default).
            verbose: Whether to print timing information.

        Returns:
            Dict with keys: ``prediction``, ``prompt``, ``timings``.
        """
        timings = {}
        output_format = output_format or self.default_format
        num_examples = num_examples or self.num_examples
        example_order = example_order or self.default_order

        # Step 1: Mark predicate in sentence
        t0 = time.time()
        words = sentence.split()
        if predicate_index < 0 or predicate_index >= len(words):
            raise ValueError(
                f"predicate_index {predicate_index} out of range for "
                f"sentence with {len(words)} words"
            )

        v_org = words[predicate_index]
        marked_words = []
        for i, w in enumerate(words):
            if i == predicate_index:
                marked_words.extend(["<predicate>", w, "</predicate>"])
            else:
                marked_words.append(w)
        marked_sentence = " ".join(marked_words)

        # Look up roles from framefiles
        roles = []
        if predicate in self.framefiles:
            roles = self.framefiles[predicate]
        roles_str = "\n".join(roles)

        timings["preprocessing"] = time.time() - t0

        # Step 2: Encode query and retrieve examples
        t1 = time.time()
        query_str = sentence + " [SEP] " + v_org + " [SEP]"
        query_vector = self.encoder.encode(query_str)

        if self.selection_strategy == "mmr":
            results = select_mmr(
                self.db, query_vector, k=num_examples,
                lambda_param=self.mmr_lambda,
                candidate_pool_size=min(num_examples * 5, len(self.db.data)),
            )
        else:
            results = select_topk(self.db, query_vector, k=num_examples)

        examples = [self.train_data[idx] for _, idx in results]
        timings["retrieval"] = time.time() - t1

        # Step 3: Build prompt
        t2 = time.time()
        test_instance = {
            "sentence": marked_sentence,
            "v_org": v_org,
            "v": predicate,
            "roles": roles_str,
            "gold": "",
        }

        eos_token = self.tokenizer.eos_token or ""
        prompt = build_prompt(
            test_instance, examples, example_order,
            self.language, output_format, eos_token,
        )
        timings["prompt_building"] = time.time() - t2

        # Step 4: LLM inference
        t3 = time.time()
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )

        prediction = self.tokenizer.decode(
            outputs[0][input_ids["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        timings["llm_inference"] = time.time() - t3
        timings["total"] = sum(timings.values())

        if verbose:
            print("Timings:")
            for k, v in timings.items():
                print(f"  {k}: {v:.3f}s")

        return {
            "prediction": prediction,
            "prompt": prompt,
            "timings": timings,
        }
