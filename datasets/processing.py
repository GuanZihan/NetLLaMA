import json
import pandas as pd


with open("./datasets/TeleQnA.txt", "r") as f:
    contents = f.read()
    data = json.loads(contents)

csv_data = pd.DataFrame(data)

csv_data = csv_data.T.rename(columns={"option 1": "option_1", "option 2" : "option_2", "option 3" : "option_3", "option 4": "option_4", "option 5": "option_5"})


csv_data = csv_data.reset_index()
print(csv_data.head())

csv_data.to_csv("./datasets/TeleQnA.csv")
csv_data.to_json("./datasets/TeleQnA.json", orient="records", indent=4)