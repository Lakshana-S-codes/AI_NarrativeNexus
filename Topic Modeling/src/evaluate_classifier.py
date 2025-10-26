import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


from text_processing import preprocess_series


df = pd.read_csv("req_data\processed\20news_18828_clean.csv")
pipeline = joblib.load("models/text_classifier.pkl")

X = df["text"]
y = df["category"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = pipeline.predict(X_test)


print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=pipeline.classes_,
            yticklabels=pipeline.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - News Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print(" Confusion matrix saved as confusion_matrix.png")
