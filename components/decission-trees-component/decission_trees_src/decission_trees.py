import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.externals import joblib
import joblib
import mlflow

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
model = DecisionTreeClassifier(criterion= 'entropy', min_samples_split= 3, max_depth=4)

data = pd.read_csv(args.training_data)
X_train = data.drop(columns=['Potability'])
y_train = data['Potability']

model.fit(X_train, y_train)


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=X_train.columns.to_list(),  
                   class_names=['Not Potable', 'Potable'],
                   filled=True)



joblib.dump(model, (Path(args.model_output) / "training_model.pkl"))
#report_path = (Path(args.model_output) / 'water-decision-tree.svg')
#fig.savefig(report_path)




with mlflow.start_run(run_name="training_model_figure") as run:
    mlflow.log_figure(fig, "water-decision-tree.png")
    