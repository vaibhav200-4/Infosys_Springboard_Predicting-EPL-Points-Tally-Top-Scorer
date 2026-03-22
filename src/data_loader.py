import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    # handle mixed date formats safely
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], dayfirst=True, errors="coerce")

    df = df.sort_values("MatchDate")

    df.reset_index(drop=True, inplace=True)

    return df