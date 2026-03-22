from src.data_loader import load_data
from src.feature_engineering import create_features
from src.model import get_models
from src.predict import predict_matches
from simulation import simulate_league
from src.ttsplit import time_train_test_split
from src.evaluate import evaluate_model
from src.monte_carlo import simulate_season_mc

from sklearn.metrics import accuracy_score

import pandas as pd
import os

OUTPUT_DIR = "data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "data/raw/epl_final.csv"


def main():

    # load raw data
    df = load_data(DATA_PATH)

    # feature engineering
    df = create_features(df)

    # split dataset
    train_df, test_df = time_train_test_split(df)

    features = [
        "HalfTimeHomeGoals",
        "HalfTimeAwayGoals",
        "HomeForm5",
        "AwayForm5"
    ]

    X_train = train_df[features]
    y_train = train_df["FullTimeResult"]

    X_test = test_df[features]
    y_test = test_df["FullTimeResult"]

    # get all models
    models = get_models()

    results = []

    best_model = None
    best_acc = 0
    best_model_name = None

    # train and compare models
    for name, model in models.items():

        print("\nTraining:", name)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print("Accuracy:", acc)

        results.append((name, acc))

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name

    print("\nBest Model:", best_model_name)
    print("Best Accuracy:", best_acc)

    # save model comparison results
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    # use best model for prediction
    preds, probs = predict_matches(best_model, X_test)

    test_df["PredictedResult"] = preds

    # evaluate best model
    evaluate_model(y_test, preds)

    # deterministic league table
    table = simulate_league(test_df)

    print("\nPredicted League Table\n")
    print(table.head())

    print("\nPredicted Champion:", table.iloc[0]["Team"])

    # save predictions
    test_df.to_csv(f"{OUTPUT_DIR}/match_predictions.csv", index=False)
    table.to_csv(f"{OUTPUT_DIR}/predicted_table.csv", index=False)

    # Monte Carlo simulation
    title_probs, winners_df = simulate_season_mc(test_df, probs, n_sim=1000)

    print("\nMonte Carlo Title Probabilities\n")
    print(title_probs.head())

    # save results
    title_probs.to_csv(f"{OUTPUT_DIR}/title_probabilities.csv", index=False)

    # save simulation winners
    winners_df.to_csv(f"{OUTPUT_DIR}/simulation_winners.csv", index=False)


if __name__ == "__main__":
    main()