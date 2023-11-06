import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
import pandas as pd
#from sklearn.externals import joblib
import joblib
import numpy as np
from numpy import save

parser = argparse.ArgumentParser("logistic_score")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--score_output", type=str, help="Path of scoring output")

args = parser.parse_args()

print("Hola desde score...")

lines = [
    f"Model path: {args.model_input}",
    f"Test data path: {args.test_data}",
    f"Scoring output path: {args.score_output}",
]

for line in lines:
    print(line)

# cargar el modelo dummy desde el archivo de texto:

joblib_file = (Path(args.model_input) / "logistic_model.pkl")
model = joblib.load(joblib_file)

print("Model: ", model)

data = pd.read_csv(args.test_data)

X_test = data.drop(columns=['Potability'])
y_test = data['Potability']

# Make predictions on the test data
y_pred = model.predict(X_test)
print("y_pred: ", y_pred)

file_score = (Path(args.score_output) / "logistic_score.npy")

save(file_score, y_pred)