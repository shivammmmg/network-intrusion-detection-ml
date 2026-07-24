"""Logistic Regression Model
Trained on the validation set"""

import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ROOT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_FOLDER / "src"))

from config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE
from preprocess import load_artifact

def load_split(split_name):
    x = pd.read_parquet(PROCESSED_DIR / f"x_{split_name}.parquet")
    y = pd.read_parquet(PROCESSED_DIR / f"y_{split_name}.parquet")["label"]

    return x, y

x_training_raw, y_training = load_split("train")
x_validation_raw, y_validation = load_split("val")

preprocessor = load_artifact(ARTIFACTS_DIR / "preprocess_linear.joblib")

x_training = preprocessor.transform(x_training_raw)
x_validation = preprocessor.transform(x_validation_raw)

print("Training shape:", x_training.shape)
print("Validation shape: ", x_validation.shape)

logr_model = LogisticRegression(max_iter = 5000, random_state = RANDOM_STATE)

logr_model.fit(x_training, y_training)

predictions = logr_model.predict(x_validation)

print("\nClassification report: ")
print(classification_report(y_validation, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_validation, predictions))

logr_model_path = ARTIFACTS_DIR / "logistic_regression.joblib"
joblib.dump(logr_model, logr_model_path)

print(f"\n Model saved to: {logr_model_path}")







