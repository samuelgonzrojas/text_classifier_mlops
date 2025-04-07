import json
import joblib
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

# Cargar parámetros
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Cargar datos y modelo
data = pd.read_csv("data/processed/test.csv")
X = data["review"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["train"]["test_size"], random_state=42
)

model = joblib.load("models/text_model.pkl")

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log en MLflow
mlflow.set_experiment("Text Classification")
with mlflow.start_run(run_name="evaluate"):
    mlflow.log_metric("accuracy", accuracy)

# Guardar en metrics.json
metrics = {"accuracy": accuracy}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Accuracy: {accuracy:.4f}")
