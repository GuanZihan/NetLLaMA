# Reference: Alpaca & Vicuna

import argparse
import io
import json
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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

PROMPT_DICT = {
    "ours": """You are an intelligent languge model for networking domain.
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
""",
    "alpaca": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Input:\n{note}\n\n### Response:"
    ),
    "medalpaca": (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        "\n\n### Instruction:\n{question}\n\n### Input:\n{note}\n\n### Response:\n"
    ),
    "chat": """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: [The start of the Discharge Summary]
{note}
[The end of the Discharge Summary]
{question} ASSISTANT: 
""",
}


def get_prompt(model_name):
    if model_name in ["decapoda-research/llama-7b-hf", "chaoyi-wu/PMC_LLAMA_7B", "NousResearch/Llama-2-7b-hf", "output_GPT_NEO_2.7B"]:
        print("Using Ours+Response Prompt")
        return PROMPT_DICT["ours"] + "\nResponse: "
    # chatdoctor, alpaca ,medalpaca
    elif model_name in [
        "chavinlo/alpaca-native",
        "zl111/ChatDoctor",
    ]:
        print("Using Alpaca Prompt")
        return PROMPT_DICT["alpaca"]
    elif model_name == "medalpaca/medalpaca-7b":
        print("Using MedAlpaca Prompt")
        return PROMPT_DICT["medalpaca"]
    elif "vicuna" in model_name or "clinical-camel" in model_name:
        print("Using Vicuna Prompt")
        return PROMPT_DICT["chat"]
    else:
        print("Using Our Prompt")
        return PROMPT_DICT["ours"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    return parser.parse_args()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def extract_answer_with_regular_expression(text):
    pattern = r"The correct option is option (\d+)"
    match = re.search(pattern, text)
    if match is None:
        return "0"
    return match.group(1)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-2.7B"
    )
    max_memory = None
    if "13" in args.model_name or "camel" in args.model_name:
        if "A100" not in torch.cuda.get_device_name():
            max_memory = {0: "40GiB", 1: "48GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        device_map="auto",
        max_memory=max_memory,
    )
    model = torch.compile(model)
    prompt = get_prompt(args.model_name)
    answers = []


    if ".json" in args.input_path:
        data = jload(args.input_path)[8000:]
        tokenizer = modify_special_tokens(tokenizer)
        for sample in tqdm(data):
            for k, v in sample.items():
                if isinstance(v, str):
                    sample[k] = v.strip("\n")
            text = prompt.format_map(sample)

            tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")

            output = model.generate(
                tokens,
                max_new_tokens=400,
                num_beams=5,
                do_sample=True,
                temperature=1,
                eos_token_id=[2],
                use_cache=True,
            )

            result = tokenizer.decode(output[0], skip_special_tokens=True)
            try:
                answer = result[len(text) : result.index("</s>", len(text))].strip()
            except:
                answer = result[len(text) :].strip()
            
            option = extract_answer_with_regular_expression(answer)
            
            answers.append({"generated": eval(option)})
    with open(args.save_path, "w") as f:
        json.dump(answers, f)


if __name__ == "__main__":
    main()