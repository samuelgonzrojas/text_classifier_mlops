import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

mlflow.set_experiment("Text Classification")

df = pd.read_csv("data/processed/train.csv")
X, y = df["review"], df["sentiment"]

pipeline = Pipeline(
    [("vectorizer", CountVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
)

with mlflow.start_run():
    pipeline.fit(X, y)
    joblib.dump(pipeline, "models/text_model.pkl")
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        input_example=["this is a sample text"],
        signature=mlflow.models.infer_signature(X, pipeline.predict(X)),
    )
    mlflow.log_param("model_type", "LogisticRegression")
