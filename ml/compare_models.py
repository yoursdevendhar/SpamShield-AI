
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "ml" / "SMSSpamCollection"

data = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

data = data.drop_duplicates(subset=["message"])

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


configs = [
    {"name": "Baseline alpha=1.0", "alpha": 1.0},
    {"name": "Experiment alpha=0.5", "alpha": 0.5},
    {"name": "Experiment alpha=0.2", "alpha": 0.2},
]

for config in configs:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )),
        ("classifier", MultinomialNB(alpha=config["alpha"]))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    ))

    print("\n" + "=" * 45)
    print(config["name"])
    print("=" * 45)
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    ))


    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


    print("\n" + "=" * 45)
    print(config["name"])
    print("=" * 45)
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    ))
