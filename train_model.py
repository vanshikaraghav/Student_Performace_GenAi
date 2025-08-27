# model.py
import os
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

SAVED_MODEL_PATH = "saved_model.pkl"
FEATURES_CACHE = "students_raw.csv"

def load_ucistudent_data(cache: bool = True) -> pd.DataFrame:
    """
    Fetch UCI Student Performance dataset (id=320) using ucimlrepo.
    Returns DataFrame with features + G3 target.
    If cache=True it will save a local CSV copy for speed.
    """
    try:
        student = fetch_ucirepo(id=320)
        X = student.data.features
        y = student.data.targets["G3"]
        df = pd.concat([X, y], axis=1)
        if cache:
            df.to_csv(FEATURES_CACHE, index=False)
        return df
    except Exception as e:
        # try reading cached CSV if fetch fails
        if os.path.exists(FEATURES_CACHE):
            return pd.read_csv(FEATURES_CACHE)
        raise RuntimeError(f"Failed to fetch dataset: {e}")

def build_and_train(df: pd.DataFrame = None, save_path: str = SAVED_MODEL_PATH):
    """
    Build preprocessing + XGBoost pipeline and train on df.
    Saves pipeline to disk (joblib).
    Returns (pipeline, train_report_str)
    """
    if df is None:
        df = load_ucistudent_data()

    # Target
    if "G3" not in df.columns:
        raise ValueError("Dataset must contain 'G3' target column.")

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=["G3"]).reset_index(drop=True)

    # Features: use numeric columns and encode categorical
    X = df.drop(columns=["G3"])
    y = df["G3"].astype(float)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
           # ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),

        ],
        remainder="passthrough"  # numeric passthrough
    )

    # Pipeline with XGBoost regressor
    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("xgb", XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model
    joblib.dump(model, save_path)

    report = (
        f"Model trained and saved to {save_path}\n"
        f"Test set MSE: {mse:.4f}\n"
        f"Test set R2: {r2:.4f}\n"
        f"Num train samples: {len(X_train)}, num test samples: {len(X_test)}"
    )
    return model, report

def load_model(path: str = SAVED_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError("Trained model not found. Please run build_and_train() or train via app.")
    return joblib.load(path)

def predict_g3_from_row(model_pipeline, row: pd.Series) -> float:
    """
    Accepts a pandas Series with feature columns (same columns as dataset except G3).
    Returns predicted G3 as float.
    """
    X = pd.DataFrame([row])
    pred = model_pipeline.predict(X)[0]
    return float(pred)

if __name__ == "__main__":
    # CLI training entry
    print("Loading dataset and training model...")
    df = load_ucistudent_data()
    model, rep = build_and_train(df)
    print(rep)
