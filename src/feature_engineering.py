import pandas as pd
import numpy as np

POST_MATCH_COLS = [
    "FullTimeHomeGoals", "FullTimeAwayGoals",
    "HalfTimeHomeGoals", "HalfTimeAwayGoals", "HalfTimeResult",
    "HomeShots", "AwayShots",
    "HomeShotsOnTarget", "AwayShotsOnTarget",
    "HomeCorners", "AwayCorners",
    "HomeFouls", "AwayFouls",
    "HomeYellowCards", "AwayYellowCards",
    "HomeRedCards", "AwayRedCards",
]


def calculate_elo(df, k=20, base_rating=1500):
    teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    ratings = {team: base_rating for team in teams}
    home_elo, away_elo = [], []
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        Rh, Ra = ratings[home], ratings[away]
        home_elo.append(Rh)
        away_elo.append(Ra)
        Eh = 1 / (1 + 10 ** ((Ra - Rh) / 400))
        Ea = 1 / (1 + 10 ** ((Rh - Ra) / 400))
        if row["FullTimeResult"] == "H":
            Sh, Sa = 1, 0
        elif row["FullTimeResult"] == "A":
            Sh, Sa = 0, 1
        else:
            Sh, Sa = 0.5, 0.5
        ratings[home] += k * (Sh - Eh)
        ratings[away] += k * (Sa - Ea)
    df = df.copy()
    df["HomeELO"]  = home_elo
    df["AwayELO"]  = away_elo
    df["ELO_diff"] = df["HomeELO"] - df["AwayELO"]
    return df


def create_form_features(df, window=5):
    df = df.copy().reset_index(drop=True)
    df["HomePoints"] = df["FullTimeResult"].map({"H": 3, "D": 1, "A": 0})
    df["AwayPoints"] = df["FullTimeResult"].map({"H": 0, "D": 1, "A": 3})
    team_pts: dict = {}
    home_form, away_form = [], []
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hist_h = team_pts.get(home, [])
        home_form.append(np.mean(hist_h[-window:]) if hist_h else np.nan)
        hist_a = team_pts.get(away, [])
        away_form.append(np.mean(hist_a[-window:]) if hist_a else np.nan)
        team_pts.setdefault(home, []).append(row["HomePoints"])
        team_pts.setdefault(away, []).append(row["AwayPoints"])
    df["HomeFormPts"] = home_form
    df["AwayFormPts"] = away_form
    df["FormDiff"]    = df["HomeFormPts"] - df["AwayFormPts"]
    return df


def add_win_rate(df, window=10):
    df = df.copy().reset_index(drop=True)
    team_wins: dict = {}
    home_wr, away_wr = [], []
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hist_h = team_wins.get(home, [])
        home_wr.append(np.mean(hist_h[-window:]) if hist_h else np.nan)
        hist_a = team_wins.get(away, [])
        away_wr.append(np.mean(hist_a[-window:]) if hist_a else np.nan)
        home_won = 1 if row["FullTimeResult"] == "H" else 0
        away_won = 1 if row["FullTimeResult"] == "A" else 0
        team_wins.setdefault(home, []).append(home_won)
        team_wins.setdefault(away, []).append(away_won)
    df["HomeWinRate"] = home_wr
    df["AwayWinRate"] = away_wr
    df["WinRateDiff"] = df["HomeWinRate"] - df["AwayWinRate"]
    return df


def add_draw_rate(df, window=10):
    """
    NEW: Rolling draw rate per team.
    Teams that draw frequently are more likely to draw next match.
    This directly addresses the model's inability to predict draws.
    """
    df = df.copy().reset_index(drop=True)
    team_draws: dict = {}
    home_dr, away_dr = [], []
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hist_h = team_draws.get(home, [])
        home_dr.append(np.mean(hist_h[-window:]) if hist_h else np.nan)
        hist_a = team_draws.get(away, [])
        away_dr.append(np.mean(hist_a[-window:]) if hist_a else np.nan)
        drew = 1 if row["FullTimeResult"] == "D" else 0
        team_draws.setdefault(home, []).append(drew)
        team_draws.setdefault(away, []).append(drew)
    df["HomeDrawRate"] = home_dr
    df["AwayDrawRate"] = away_dr
    # Both teams draw a lot → draw more likely
    df["CombinedDrawRate"] = df["HomeDrawRate"] + df["AwayDrawRate"]
    return df


def add_rest_days(df):
    df = df.copy().reset_index(drop=True)
    last_match: dict = {}
    home_rest, away_rest = [], []
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        date = row["MatchDate"]
        home_rest.append((date - last_match[home]).days if home in last_match else np.nan)
        away_rest.append((date - last_match[away]).days if away in last_match else np.nan)
        last_match[home] = date
        last_match[away] = date
    df["HomeRestDays"] = home_rest
    df["AwayRestDays"] = away_rest
    df["RestDiff"]     = df["HomeRestDays"] - df["AwayRestDays"]
    df[["HomeRestDays", "AwayRestDays", "RestDiff"]] = (
        df[["HomeRestDays", "AwayRestDays", "RestDiff"]].fillna(7)
    )
    return df


def add_head_to_head(df, window=5):
    df = df.copy()
    df["H2H_home_win_rate"] = np.nan
    df["H2H_draw_rate"]     = np.nan
    history: dict = {}
    for idx, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key = tuple(sorted([home, away]))
        past = history.get(key, [])[-window:]
        if past:
            df.loc[idx, "H2H_home_win_rate"] = np.mean([r for r in past])
            df.loc[idx, "H2H_draw_rate"]     = np.mean([1 if r == 0.5 else 0 for r in past])
        result = 1 if row["FullTimeResult"] == "H" else (
                 0 if row["FullTimeResult"] == "A" else 0.5)
        history.setdefault(key, []).append(result)
    return df


def add_goal_diff_rolling(df, window=5):
    df = df.copy().reset_index(drop=True)
    team_gd: dict = {}
    home_gd, away_gd = [], []
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hist_h = team_gd.get(home, [])
        home_gd.append(np.mean(hist_h[-window:]) if hist_h else np.nan)
        hist_a = team_gd.get(away, [])
        away_gd.append(np.mean(hist_a[-window:]) if hist_a else np.nan)
        hg = row["FullTimeHomeGoals"]
        ag = row["FullTimeAwayGoals"]
        team_gd.setdefault(home, []).append(hg - ag)
        team_gd.setdefault(away, []).append(ag - hg)
    df["HomeAvgGD"] = home_gd
    df["AwayAvgGD"] = away_gd
    df["AvgGDDiff"] = df["HomeAvgGD"] - df["AwayAvgGD"]
    return df


def add_draw_sensitive_features(df):
    """
    NEW: Features specifically designed to signal draw likelihood.

    Key insight: draws happen when teams are evenly matched.
    ELO closeness + both teams scoring similarly + both defensive → draw likely.
    """
    df = df.copy()

    # --- Closeness indicators ---
    # Small ELO gap → evenly matched → more likely to draw
    df["AbsELODiff"]  = df["ELO_diff"].abs()
    df["CloseMatch"]  = (df["AbsELODiff"] < 50).astype(int)

    # Small form gap → neither team dominant right now
    df["AbsFormDiff"] = df["FormDiff"].abs()

    # Small avg GD → both teams low-scoring → draw territory
    df["AbsAvgGDDiff"]    = df["AvgGDDiff"].abs()
    df["BothLowScoring"]  = ((df["HomeAvgGD"].abs() < 0.5) &
                              (df["AvgGDDiff"].abs() < 0.5)).astype(int)

    # Combined defensive strength → low goals → draws
    df["AvgGoalDiff"]     = (df["HomeAvgGD"] + df["AwayAvgGD"].abs()) / 2

    return df


def preprocess_features(df):
    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df = df.sort_values("MatchDate").reset_index(drop=True)

    df = calculate_elo(df)
    df = add_goal_diff_rolling(df)
    df = create_form_features(df)
    df = add_win_rate(df)
    df = add_draw_rate(df)          # NEW
    df = add_rest_days(df)
    df = add_head_to_head(df)
    df = add_draw_sensitive_features(df)   # NEW — must come after ELO/form/GD

    df["Target"] = df["FullTimeResult"].map({"H": 2, "D": 1, "A": 0})

    cols_to_drop = [c for c in POST_MATCH_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    for col in ["HomeTeam", "AwayTeam", "MatchDate"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    remaining_leaky = [c for c in POST_MATCH_COLS if c in df.columns]
    if remaining_leaky:
        raise ValueError(f"LEAKAGE DETECTED: {remaining_leaky}")

    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c not in ("Target", "Season", "FullTimeResult")]
    print(f"\n[feature_engineering] {len(feature_cols)} features: {feature_cols}")
    print(f"[feature_engineering] Shape: {df.shape}\n")

    return df