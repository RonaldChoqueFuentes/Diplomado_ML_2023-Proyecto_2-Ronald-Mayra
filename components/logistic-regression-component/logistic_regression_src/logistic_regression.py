import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
from sklearn.linear_model import LogisticRegression
import pandas as pd
#from sklearn.externals import joblib
import joblib
# obtener parámetros:
parser = argparse.ArgumentParser("logistic_regression")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

lines = [
    f"Training data path: {args.training_data}",
    f"Model output path: {args.model_output}",
]

print("Parametros: ...")

# imprimir parámetros:

for line in lines:
    print(line)
    

# Create a binary classification model (Logistic Regression in this case)
model = LogisticRegression()

data = pd.read_csv(args.training_data)
X_train = data.drop(columns=['Potability'])
y_train = data['Potability']

model.fit(X_train, y_train)


joblib.dump(model, (Path(args.model_output) / "training_model.pkl"))

