import numpy as np

def ensemble_predict(models, X):

    probs = []

    for model in models:
        probs.append(model.predict_proba(X))

    probs = np.mean(probs, axis=0)

    preds = np.argmax(probs, axis=1)

    return preds, probs