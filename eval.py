# Reference: Alpaca & Vicuna

import argparse
import io
import json
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

import time

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
"""
}


# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append((
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest',).to("cuda"),
            prompt_batch)
            )
    tokenizer.padding_side="right"
    return batches_tok

def get_prompt():
    print("Using Ours+Response Prompt")
    return PROMPT_DICT["ours"]

def write_pretty_json(file_path, data):
    import json
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--batch_size", type=int, default=16)

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
        args.tokenizer_name
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
    prompt = get_prompt()
    answers = []


    prompts = []

    



    if ".json" in args.input_path:
        data = jload(args.input_path)[9990:]
        tokenizer = modify_special_tokens(tokenizer)
        for sample in tqdm(data):
            for k, v in sample.items():
                if isinstance(v, str):
                    sample[k] = v.strip("\n")
            text = prompt.format_map(sample)
            prompts.append(text)
        
    accelerator.wait_for_everyone()
    start=time.time()
    with accelerator.split_between_processes(prompts) as prompts:
        results=dict(outputs=[], num_tokens=0, prompts=[], option =[])
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.batch_size)

        for prompts_tokenized, original_prompt in tqdm(prompt_batches):
            outputs_tokenized = model.generate(
                **prompts_tokenized,
                max_new_tokens=400,
                num_beams=5,
                do_sample=True,
                temperature=1,
                eos_token_id=[2],
                use_cache=True,
            )
        
            # remove prompt from gen. tokens
            outputs_tokenized=[tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ]
            # count and decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized)

            num_tokens=sum([ len(t) for t in outputs_tokenized ])

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["prompts"].extend(original_prompt)
            results["num_tokens"] += num_tokens
            
            results["option"].extend([eval(extract_answer_with_regular_expression(o)) for o in outputs])
        results=[ results ]

    results_gathered=gather_object(results)
    write_pretty_json(args.save_path, results_gathered)

    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])

        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

        # result = tokenizer.decode(output[0], skip_special_tokens=True)
        # try:
        #     answer = result[len(text) : result.index("</s>", len(text))].strip()
        # except:
        #     answer = result[len(text) :].strip()
        # print(text)
        # print("-------------------")
        # print(answer)
    #     option = extract_answer_with_regular_expression(answer)
        
    #     answers.append({"generated": eval(option)})
    # with open(args.save_path, "w") as f:
    #     json.dump(answers, f)

if __name__ == "__main__":
    accelerator = Accelerator()
    main()