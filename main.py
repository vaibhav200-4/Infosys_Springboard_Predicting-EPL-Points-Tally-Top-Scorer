"""
main.py — EPL match outcome prediction pipeline.

Run: python main.py
"""

from src.data_loader import load_data
from src.feature_engineering import preprocess_features
from src.ttsplit import split_data
from src.tuning import tune_xgboost, tune_lightgbm, tune_random_forest
from src.evaluate import evaluate_model


def main():
    # 1. Load raw data
    df = load_data("data/raw/epl_final.csv")

    # 2. Feature engineering (must happen before split — ELO/form needs full history)
    df = preprocess_features(df)

    # 3. Chronological train/test split (Season >= 2022-23 → test)
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Train size : {len(X_train):,} matches")
    print(f"Test size  : {len(X_test):,} matches")
    print(f"Features   : {X_train.shape[1]}")

    # 4. Tune and evaluate — try XGBoost first (typically best for this task)
    print("\n=== XGBoost ===")
    xgb_model = tune_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)

    print("\n=== LightGBM ===")
    lgbm_model = tune_lightgbm(X_train, y_train)
    evaluate_model(lgbm_model, X_test, y_test)

    print("\n=== Random Forest ===")
    rf_model = tune_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)


if __name__ == "__main__":
    main()