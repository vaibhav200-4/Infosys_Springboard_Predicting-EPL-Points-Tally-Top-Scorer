"""
model.py — Model definitions for EPL match outcome prediction.

EPL prediction is a 3-class problem (H / D / A). A well-calibrated
gradient boosting model typically outperforms Random Forest for this task.
We expose both so tuning.py can choose.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_random_forest(n_estimators=300, max_depth=8, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgboost(random_state=42):
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def get_lightgbm(random_state=42):
    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_ensemble(random_state=42):
    """Soft-voting ensemble — averages predicted probabilities."""
    return VotingClassifier(
        estimators=[
            ("xgb",  get_xgboost(random_state)),
            ("lgbm", get_lightgbm(random_state)),
            ("rf",   get_random_forest(random_state=random_state)),
        ],
        voting="soft",
        n_jobs=-1,
    )