import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# obtener parámetros:
parser = argparse.ArgumentParser("split")
parser.add_argument("--clean_data", type=str, help="Path of the clean data")
parser.add_argument("--training_data", type=str, help="Path of output training data")
parser.add_argument("--testing_data", type=str, help="Path of output testing data")

args = parser.parse_args()

print("Hola desde train...")

lines = [
    f"clean_data: {args.clean_data}",
    f"training_data: {args.training_data}",
    f"testing_data: {args.testing_data}",
]

# imprimir parámetros:

for line in lines:
    print(line)

data = pd.read_csv(args.clean_data)

X = data.drop(columns=['Potability'])
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


training_data = pd.concat([X_train,y_train],axis=1)

testing_data = pd.concat([X_test,y_test],axis=1)

training_data.to_csv(args.training_data)
testing_data.to_csv(args.testing_data)