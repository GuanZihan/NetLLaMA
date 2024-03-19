from utils import tokenize_data
import datasets
import os
from transformers import AutoTokenizer
import argparse


parser = argparse.ArgumentParser(
                    prog='ProcessingData')

parser.add_argument('--input_file', type=str, default="./datasets/TeleQnA.csv")
parser.add_argument('--model_name', type=str, default="EleutherAI/gpt-neo-2.7B")
parser.add_argument("--save_path", type=str, default="./TeleQnA")

args = parser.parse_args()


if os.path.exists(args.save_path):
    dataset = datasets.load_from_disk(args.save_path)
else:
    dataset = tokenize_data(input_path=args.input_file, model_name=args.model_name, save_path=args.save_path)