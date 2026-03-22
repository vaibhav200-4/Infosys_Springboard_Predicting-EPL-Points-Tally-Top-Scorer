import os
import pandas as pd

def split_data(df):

    # season wise split
    train = df[df["Season"] < "2022-2023"]
    test = df[df["Season"] >= "2022-2023"]

    # split features and target
    X_train = train.drop(columns=["Target","FullTimeResult","Season","HomePoints", "AwayPoints"])
    y_train = train["Target"]

    X_test = test.drop(columns=["Target","FullTimeResult","Season","HomePoints", "AwayPoints"])
    y_test = test["Target"]

    # create processed directory
    os.makedirs("data/processed", exist_ok=True)

    # save clean feature datasets
    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)

    # optional: save targets separately
    y_train.to_csv("data/processed/train_target.csv", index=False)
    y_test.to_csv("data/processed/test_target.csv", index=False)

    return X_train, X_test, y_train, y_test