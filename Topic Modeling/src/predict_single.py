import joblib
from text_processing import preprocess_series

pipeline = joblib.load("models/text_classifier.pkl")

examples = [
    "The new graphics card from NVIDIA has amazing performance for 3D rendering.",
    "God does not exist, and religion is just a human creation.",
    "The new Windows update has caused issues with drivers.",
    "NASA discovered water on Mars, confirming planetary research findings."
]

predictions = pipeline.predict(examples)

for text, label in zip(examples, predictions):
    print(f"\n Text: {text}\n Predicted Category: {label}")
print("\n Predictions complete.")