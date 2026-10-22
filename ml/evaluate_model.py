
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Find project files
project_dir = Path(__file__).resolve().parent.parent

data_path = project_dir / "ml" / "SMSSpamCollection"
model_path = project_dir / "backend" / "spam_model.joblib"

# Load dataset
data = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Use the same data preparation and split as training
data = data.drop_duplicates(subset=["message"])

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

# Load the existing saved model
model = joblib.load(model_path)

# Predict the test messages
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Precision, recall and F1-score
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)

# Confusion matrix
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"]
)

print("Confusion Matrix:")
print(matrix)

print("\nEvaluation completed!")
