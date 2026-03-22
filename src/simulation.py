import pandas as pd
import numpy as np


def simulate_season(matches, model, features, n_simulations=1000):

    teams = pd.unique(matches[["HomeTeam","AwayTeam"]].values.ravel())

    champion_count = {team:0 for team in teams}

    for sim in range(n_simulations):

        points = {team:0 for team in teams}

        for _, row in matches.iterrows():

            home = row["HomeTeam"]
            away = row["AwayTeam"]

            X = row[features].to_frame().T

            probs = model.predict_proba(X)[0]

            result = np.random.choice([0,1,2], p=probs)

            if result == 2:
                points[home] += 3
            elif result == 0:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

        champion = max(points, key=points.get)
        champion_count[champion] += 1

    results = pd.DataFrame({
        "Team": champion_count.keys(),
        "ChampionProbability": [v/n_simulations for v in champion_count.values()]
    })

    return results.sort_values("ChampionProbability", ascending=False)