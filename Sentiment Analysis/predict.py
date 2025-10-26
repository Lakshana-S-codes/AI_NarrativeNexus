import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train_test import clean_text

# Load saved RF + TF-IDF
rf_loaded = joblib.load("random_forest_model.pkl")
vectorizer_loaded = joblib.load("tfidf_vectorizer.pkl")

def predict_rf(text):
    text_clean = clean_text(text)
    vec = vectorizer_loaded.transform([text_clean])
    pred = rf_loaded.predict(vec)[0]
    return "Positive " if pred == 1 else "Negative "


lstm_loaded = load_model("lstm_model.h5")
tokenizer_loaded = joblib.load("lstm_tokenizer.pkl")

def predict_lstm(text):
    text_clean = clean_text(text)
    seq = tokenizer_loaded.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=200)
    pred = (lstm_loaded.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive " if pred == 1 else "Negative "

sample_texts = [
    "This product is absolutely fantastic, I loved it!",
    "Worst purchase ever, total waste of money.",
    "It was okay, nothing special but not bad either."
]

for txt in sample_texts:
    print(f"\nINPUT: {txt}")
    print("Random Forest →", predict_rf(txt))
    print("LSTM          →", predict_lstm(txt))

