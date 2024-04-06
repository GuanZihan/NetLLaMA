python eval.py --model_name output_gpt-neo-1.3B \
     --input_path ./datasets/TeleQnA.json \
     --save_path ./save/all_answers_output_gpt-neo-1.3B.json \
     --tokenizer_name EleutherAI/gpt-neo-1.3B \
     --batch_size 16

python eval.py --model_name output_Llama-2-7b-hf \
     --input_path ./datasets/TeleQnA.json \
     --save_path ./save/all_answers_output_Llama-2-7b-hf.json \
     --tokenizer_name NousResearch/Llama-2-7b-hf \
     --batch_size 16