# Adapted From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py


#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import io
import json
import logging
from typing import Dict, Sequence, Optional
import datasets
from dataclasses import field, dataclass
import pandas as pd

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer


# if "A100" in torch.cuda.get_device_name():
#     from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

#     replace_llama_attn_with_flash_attn()


IGNORE_INDEX = -100

PROMPT = """You are an intelligent languge model for networking domain.
Below is a choice question related to networking domain. Please carefully read the questions and generate the answers.

[Question Begin]
{question}
[Question End]

[Option Begin]
Option 1: {option_1}
Option 2: {option_2}
Option 3: {option_3}
Option 4: {option_4}
[Option End]
"""


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        if ".json" in data_path:
            list_data_dict = jload(data_path)
            list_data_dict = list_data_dict[:int(0.8 * len(list_data_dict))]
            print(len(list_data_dict))
            for i in range(len(list_data_dict)):
                for k, v in list_data_dict[i].items():
                    if isinstance(v, str):
                        list_data_dict[i][k] = v.strip("\n")
            logging.warning("Formatting inputs...")
        else:
            raise NotImplementedError

        sources = [PROMPT.format_map(example) for example in list_data_dict]

        targets = [
            "The correct option is {}.\nTherefore the correct answer is {}.\nExplanation for my answer is: {}{}".format(example['answer'].split(":")[0], example['answer'].split(":")[1], example['explanation'], tokenizer.eos_token) for example in list_data_dict
        ]



        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print("====Finish Preparing Dataset====")
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
    )
    dataloader_num_workers: int = field(
        default=16,
    )

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer = modify_special_tokens(tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()