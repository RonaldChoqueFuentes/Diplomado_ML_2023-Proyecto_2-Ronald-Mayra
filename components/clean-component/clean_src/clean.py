import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd

# obtener parámetros:
parser = argparse.ArgumentParser("clean")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--model_output", type=str, help="Path of output clean data")

args = parser.parse_args()

print("Hola desde train...")

lines = [
    f"Training data path: {args.training_data}",
    f"Model output path: {args.model_output}",
]

# imprimir parámetros:

for line in lines:
    print(line)

data = pd.read_csv(args.training_data)
data = data.fillna(data.mean())

print(data.isnull().sum())

data.to_csv(args.model_output)