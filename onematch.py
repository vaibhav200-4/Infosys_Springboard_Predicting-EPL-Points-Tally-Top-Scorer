import joblib
import pandas as pd

model = joblib.load("models/epl_model.pkl")
features = joblib.load("models/features.pkl")

sample = pd.read_csv("data/processed/test.csv").iloc[[0]]

# ensure correct feature order
sample = sample[features]

pred = model.predict(sample)[0]

mapping = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win"
}

print("Predicted Result:", mapping[pred])