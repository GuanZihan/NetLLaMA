import json
import pandas as pd
import numpy as np
with open("./save/all_answers_output_gpt-neo-1.3B.json", "r") as f:
    file = json.load(f)
    all_answers = pd.DataFrame(file)

all_ground_truth = []
with open("./datasets/TeleQnA.json", "r") as f:
    file = json.load(f)
    ground_truth = pd.DataFrame(file)
    for id, item in ground_truth["answer"].items():
        if id >= 8000:
            all_ground_truth.append(eval(item.split(":")[0].split(" ")[1]))


print(sum(np.array(all_ground_truth) == np.array(all_answers.iloc[:, -1][0])))