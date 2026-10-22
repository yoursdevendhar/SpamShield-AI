from pathlib import Path
import joblib

project_dir = Path(__file__).resolve().parent.parent

model_path = project_dir/"backend"/"spam_model.joblib"
model = joblib.load(model_path)


test_messages = [
    "Congratulations! You won a free prize. Click here now!",
    "Hey, are we meeting for lunch today?",
    "URGENT! Claim your free cash reward now!",
    "Can you send me the notes from today's class?",
    "You have been selected for a free vacation. Call now!",
    "Are you coming home tonight?",
]

for message in test_messages:
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_

    confidence = float(max(probabilities) * 100)
    warning = None

    if confidence < 70:
        warning = "Low confidence: this prediction may be unreliable."

    print("\nMessage:", message)
    print("Prediction:", prediction)
    for class_name, probability in zip(classes, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")
