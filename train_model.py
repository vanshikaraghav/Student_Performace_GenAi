"""
train_model.py

Train and save an XGBoost classifier and the LabelEncoder.
This uses a synthetic dataset by default but accepts a DataFrame input
if you want to train on real data.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

MODEL_PATH = "saved_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

def generate_synthetic_data(n=500, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame({
        "StudyHours": np.random.randint(1, 13, n),               # 1-12 hours
        "Attendance": np.random.randint(50, 101, n),            # 50-100%
        "AssignmentsCompleted": np.random.randint(0, 11, n),    # 0-10
        # skew distribution: more Medium/High to look plausible
        "PerformanceLevel": np.random.choice(["High", "Medium", "Low"], n, p=[0.35, 0.45, 0.20])
    })
    return df

def train_and_save(df=None, save_model_path=MODEL_PATH, save_encoder_path=ENCODER_PATH):
    """
    Train model on df (if None, synthetic data is generated).
    Saves model and encoder to disk.
    Returns (model, label_encoder, train_report_str)
    """
    if df is None:
        df = generate_synthetic_data()

    # Basic validation
    required = {"StudyHours", "Attendance", "AssignmentsCompleted", "PerformanceLevel"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["PerformanceLevel"])

    X = df[["StudyHours", "Attendance", "AssignmentsCompleted"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost multi-class classifier
    num_classes = len(le.classes_)
    model = XGBClassifier(
        use_label_encoder=False,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    # Save artifacts
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    joblib.dump(model, save_model_path)
    joblib.dump(le, save_encoder_path)

    summary = (
        f"Model trained and saved.\nAccuracy on test set: {acc*100:.2f}%\n\n"
        f"Label classes: {list(le.classes_)}\n\nClassification report:\n{report}"
    )
    return model, le, summary

if __name__ == "__main__":
    print("Training model on synthetic data...")
    model, le, summary = train_and_save()
    print(summary)
