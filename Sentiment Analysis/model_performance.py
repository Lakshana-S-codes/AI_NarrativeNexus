# model_performance.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import data
from train_test import X_test, y_test

# Import predictions from models
from model_1 import y_pred_rf
from model_2 import y_pred_lstm



print("Train/Test sizes:")
print("Test size:", len(X_test))
print("Test labels distribution:\n", y_test.value_counts())



# Collect metrics
results = {
    "Model": ["Random Forest", "LSTM"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_lstm),

    ],
    "Precision": [
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_lstm),

    ],
    "Recall": [
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_lstm),

    ],
    "F1-Score": [
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_lstm),

    ],
}

df_results = pd.DataFrame(results)
print(" Model Performance Summary")
print(df_results)