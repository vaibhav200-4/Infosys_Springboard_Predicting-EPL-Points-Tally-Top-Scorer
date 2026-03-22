"""
tuning.py — Hyperparameter tuning with time-series-aware cross-validation.

IMPORTANT: Standard k-fold CV shuffles data randomly, which causes leakage
in time-series problems (future data folds into past training windows).
Use TimeSeriesSplit instead — it always trains on the past and validates
on strictly later data.
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ── Random Forest ────────────────────────────────────────────────────────────

RF_PARAM_DIST = {
    "n_estimators":   [200, 300, 500],
    "max_depth":      [4, 6, 8, 10, None],
    "min_samples_leaf": [3, 5, 10, 15],
    "max_features":   ["sqrt", "log2", 0.5],
}


def tune_random_forest(X_train, y_train, n_iter=30, n_splits=5, random_state=42):
    """
    BUG FIX: replaced KFold/StratifiedKFold with TimeSeriesSplit.
    Data must be sorted chronologically before calling this (done in pipeline).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        param_distributions=RF_PARAM_DIST,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"[RF] Best CV accuracy : {search.best_score_:.4f}")
    print(f"[RF] Best params      : {search.best_params_}")
    return search.best_estimator_


# ── XGBoost ──────────────────────────────────────────────────────────────────

XGB_PARAM_DIST = {
    "n_estimators":      [300, 500, 700],
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "subsample":         [0.7, 0.8, 0.9],
    "colsample_bytree":  [0.7, 0.8, 0.9],
    "min_child_weight":  [1, 3, 5],
    "gamma":             [0, 0.1, 0.3],
}


def tune_xgboost(X_train, y_train, n_iter=30, n_splits=5, random_state=42):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"[XGB] Best CV accuracy : {search.best_score_:.4f}")
    print(f"[XGB] Best params      : {search.best_params_}")
    return search.best_estimator_


# ── LightGBM ─────────────────────────────────────────────────────────────────

LGBM_PARAM_DIST = {
    "n_estimators":    [300, 500, 700],
    "max_depth":       [4, 5, 6, 8],
    "num_leaves":      [15, 31, 63],
    "learning_rate":   [0.01, 0.03, 0.05, 0.1],
    "subsample":       [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_samples": [10, 20, 30],
}


def tune_lightgbm(X_train, y_train, n_iter=30, n_splits=5, random_state=42):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        param_distributions=LGBM_PARAM_DIST,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"[LGBM] Best CV accuracy : {search.best_score_:.4f}")
    print(f"[LGBM] Best params      : {search.best_params_}")
    return search.best_estimator_