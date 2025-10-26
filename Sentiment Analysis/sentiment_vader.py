# Install VADER if not already installed

import nltk
nltk.download("punkt")
nltk.download('punkt_tab') # Add this line to download the missing resource
from train_test import train_df, test_df


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Function to compute sentence-level sentiment
def get_sentiment_sentence_level(text):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)  # split into sentences
    if not sentences:
        return 1  # default to positive if empty

    # Compute compound score for each sentence
    scores = [analyzer.polarity_scores(sent)["compound"] for sent in sentences]
    avg_score = sum(scores)/len(scores)

    # Convert average compound score to binary label
    return 1 if avg_score >= 0 else 0

# Apply sentence-level sentiment to preprocessed text
train_df["pred_label"] = train_df["text"].apply(get_sentiment_sentence_level)
test_df["pred_label"]  = test_df["text"].apply(get_sentiment_sentence_level)

# Evaluate
print("Train Accuracy:", accuracy_score(train_df["label"], train_df["pred_label"]))
print("Test Accuracy:", accuracy_score(test_df["label"], test_df["pred_label"]))

print("\nTest Classification Report:\n")
print(classification_report(test_df["label"], test_df["pred_label"]))