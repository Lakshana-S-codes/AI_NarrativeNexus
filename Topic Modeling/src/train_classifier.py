# training/train_classifier.py

import os
import pandas as pd
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

INPUT_CSV = "req_data\processed\20news_18828_clean.csv"
MODEL_PATH = os.path.join("models", "text_classifier.pkl")
CONF_MATRIX_PATH = os.path.join("models", "confusion_matrix.png")

# ===================
# 1. Load Dataset & Inspect Columns
# ===================
print(f"📂 Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print(f"📋 Available columns: {list(df.columns)}")
print(f"📊 Dataset shape: {df.shape}")
print(f"📝 First few rows:")
print(df.head())

# ===================
# 2. Identify Text and Target Columns
# ===================
# Common column names for text data
possible_text_cols = ['clean_text', 'text', 'content', 'message', 'data', 'body']
possible_target_cols = ['category', 'target', 'label', 'class', 'newsgroup']

text_col = None
target_col = None

# Find text column
for col in possible_text_cols:
    if col in df.columns:
        text_col = col
        break

# Find target column  
for col in possible_target_cols:
    if col in df.columns:
        target_col = col
        break

# If not found, use first text-like column
if text_col is None:
    # Look for columns with string data that might be text
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.len().mean() > 50:  # Assume text if avg length > 50
            text_col = col
            break

# If target not found, use last column or ask user
if target_col is None:
    if len(df.columns) >= 2:
        target_col = df.columns[-1]  # Assume last column is target
    else:
        raise ValueError("Could not identify target column. Please specify manually.")

if text_col is None:
    raise ValueError("Could not identify text column. Please specify manually.")

print(f"🎯 Using text column: '{text_col}'")
print(f"🏷️  Using target column: '{target_col}'")

# ===================
# 3. Data Cleaning
# ===================
# Drop rows with missing or empty text
initial_rows = len(df)
df = df.dropna(subset=[text_col, target_col])
df = df[df[text_col].str.strip() != ""]

print(f"🧹 Dropped {initial_rows - len(df)} rows with missing/empty data")

# Drop categories with <2 samples (needed for stratified split)
initial_categories = df[target_col].nunique()
df = df.groupby(target_col).filter(lambda x: len(x) > 1)
final_categories = df[target_col].nunique()

print(f"📊 Categories: {initial_categories} → {final_categories}")

if len(df) == 0:
    raise ValueError("No data left after cleaning!")

X = df[text_col]
y = df[target_col]

print(f"📊 Final dataset: {len(df)} rows, {y.nunique()} categories")
print(f"📈 Category distribution:")
print(y.value_counts().head(10))

# ===================
# 4. Train-Test Split
# ===================
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train set: {len(X_train)}, Test set: {len(X_test)}")
except ValueError as e:
    print(f"⚠️  Stratified split failed: {e}")
    print("Using random split instead...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

# ===================
# 5. Pipeline
# ===================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english')),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

print("🚀 Training pipeline...")
pipeline.fit(X_train, y_train)

# ===================
# 6. Evaluation
# ===================
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ===================
# 7. Confusion Matrix
# ===================
try:
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    
    # Determine figure size based on number of classes
    n_classes = len(pipeline.classes_)
    fig_size = min(max(n_classes * 0.8, 8), 20)  # Scale with classes but cap at 20
    
    plt.figure(figsize=(fig_size, fig_size))
    
    # Show numbers only if not too many classes
    show_annot = n_classes <= 15
    
    sns.heatmap(cm, annot=show_annot, cmap="Blues", fmt='d',
                xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    plt.savefig(CONF_MATRIX_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {CONF_MATRIX_PATH}")
    
except Exception as e:
    print(f"⚠️  Could not create confusion matrix: {e}")

# ===================
# 8. Save Model
# ===================
try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    
    # Save column mapping for future reference
    column_info = {
        'text_column': text_col,
        'target_column': target_col,
        'classes': list(pipeline.classes_),
        'accuracy': accuracy
    }
    
    info_path = os.path.join("models", "model_info.txt")
    with open(info_path, 'w') as f:
        for key, value in column_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📋 Model info saved to {info_path}")
    
except Exception as e:
    print(f"❌ Error saving model: {e}")

print(f"\n🎉 Training completed successfully!")
print(f"📈 Final accuracy: {accuracy:.4f}")
print(f"🏷️  Number of classes: {len(pipeline.classes_)}")