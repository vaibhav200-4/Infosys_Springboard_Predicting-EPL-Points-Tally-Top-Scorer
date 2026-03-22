import numpy as np
import pandas as pd


def simulate_season_mc(df, probs, n_sim=1000):

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())

    title_counts = {team: 0 for team in teams}

    winners = []

    for _ in range(n_sim):

        points = {team: 0 for team in teams}

        for idx, row in enumerate(df.itertuples()):

            home = row.HomeTeam
            away = row.AwayTeam

            p_home, p_draw, p_away = probs[idx]

            result = np.random.choice(
                ["H", "D", "A"],
                p=[p_home, p_draw, p_away]
            )

            if result == "H":
                points[home] += 3
            elif result == "A":
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

        winner = max(points, key=points.get)

        winners.append(winner)

        title_counts[winner] += 1

    # probability table
    probs_table = pd.DataFrame(
        [(team, count / n_sim) for team, count in title_counts.items()],
        columns=["Team", "TitleProbability"]
    ).sort_values("TitleProbability", ascending=False)

    # winners dataframe
    winners_df = pd.DataFrame({"Winner": winners})

    return probs_table, winners_df