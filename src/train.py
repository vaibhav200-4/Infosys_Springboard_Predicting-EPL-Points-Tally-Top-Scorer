import joblib
import os

def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/epl_model.pkl")
    joblib.dump(X_train.columns.tolist(), "models/features.pkl")

    print("Model saved")

    return model