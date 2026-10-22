from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir/"ml"/"SMSSpamCollection"
data = pd.read_csv(
    data_path,
    sep = "\t",
    header=None,
    names = ["label","message"],
    encoding="utf-8"
)
print(len(data))
print(data["label"].value_counts())

data = data.drop_duplicates(subset=["message"])
print(len(data))
X = data['message']
y = data["label"]
print(X.head())
print(y.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

# Create a machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", MultinomialNB(alpha=0.2))
])
# Train the model using the training messages
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions on unseen messages
predictions = model.predict(X_test)

print("Predictions completed!")
print(predictions[:10])
print("Actual labels:")
print(y_test.iloc[:10].to_list())

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

print("\nConfusion Matrix:")

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"]
)

print(matrix)

model_path = project_dir / "backend" / "spam_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)
