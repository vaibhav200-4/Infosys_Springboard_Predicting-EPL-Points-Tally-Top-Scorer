import joblib

def load_model(path="models/match_model.pkl"):

    return joblib.load(path)


def predict_matches(model, X):

    preds = model.predict(X)

    probs = model.predict_proba(X)

    return preds, probs