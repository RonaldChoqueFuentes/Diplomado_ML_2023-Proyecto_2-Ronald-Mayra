import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from numpy import load
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("logistic_eval")
parser.add_argument("--scoring_result", type=str, help="Path of scoring result")
parser.add_argument("--test_data", type=str, help="Path of test data")
parser.add_argument("--eval_output", type=str, help="Path of output evaluation result")

args = parser.parse_args()

print("Hola desde eval...")

lines = [
    f"Scoring result path: {args.scoring_result}",
    f"Evaluation output path: {args.eval_output}",
    f"test_data: {args.test_data}",
]

for line in lines:
    print(line)

file_score = (Path(args.scoring_result) / "score.npy")
y_pred = load(file_score)


print(y_pred)

data = pd.read_csv(args.test_data)

X_test = data.drop(columns=['Potability'])
y_test = data['Potability']


output = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
df_rep = pd.DataFrame(output).transpose()
print(df_rep)

df_rep.to_csv((Path(args.eval_output) / "eval_result.csv"))
