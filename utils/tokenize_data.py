# Adapted From: https://raw.githubusercontent.com/zphang/minimal-llama/main/tokenize_dataset.py
import re

import datasets
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from pandarallel import pandarallel
from transformers import AutoTokenizer

def modify_special_tokens(tokenizer):
    tokenizer.add_special_tokens(
        {
            "pad_token": "<s>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        }
    )

    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.unk_token_id = 0
    tokenizer.pad_token_id = 1

    return tokenizer

def tokenize_data(input_path="", model_name="baffo32/decapoda-research-llama-7B-hf", save_path=""):
    pandarallel.initialize(nb_workers=32, progress_bar=True)
    revision = "pr/7" if "decapoda-research/llama" in model_name else "main"
    print(model_name)

    use_fast = True if "pythia" in model_name else False
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast
    )
    tokenizer = modify_special_tokens(tokenizer)
    df = pd.read_csv(input_path)

    all_tokenized = []
    text_name = ['question', 'option_1', 'option_2', 'option_3', 'option_4']

    all_tokenized = df[text_name].sample(frac=1).parallel_apply(tokenizer.encode)
    print(f"Total number of tokens if {all_tokenized.str.len().sum()}")

    all_tokens = [1] + [
        tok
        for row in all_tokenized
        for tok in row + [tokenizer.eos_token_id, tokenizer.bos_token_id]
    ]

    truncated_tokens = all_tokens[: (len(all_tokens) // 2048) * 2048]
    arr = np.array(truncated_tokens).reshape(-1, 2048)
    ds = datasets.Dataset.from_dict({"input_ids": arr})

    ds.save_to_disk(save_path)
    print(f"Generated {arr.shape[0]} samples.")
    return ds