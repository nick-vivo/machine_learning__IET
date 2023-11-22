import pandas as pd

data = pd.read_csv("input.txt", sep=",")

data_2 = data["feature_1"] + data["feature_2"] 

file_path = "output.txt"

with open(file_path, "w") as file:
    for item in data_2:
        file.write(str(item) + "\n")
