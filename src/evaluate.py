from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score

model = load("models/text_model.pkl")
df = pd.read_csv("data/processed/test.csv")

preds = model.predict(df["review"])
acc = accuracy_score(df["sentiment"], preds)

print(f"Accuracy: {acc}")
