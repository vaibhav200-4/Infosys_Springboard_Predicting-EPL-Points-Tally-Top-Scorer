import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.model import get_models
from src.train import train_model
from src.predict import predict_matches
from simulation import simulate_league
from src.ttsplit import time_train_test_split
from src.evaluate import evaluate_model
from src.monte_carlo import simulate_season_mc


DATA_PATH = "data/raw/epl_final.csv"

st.set_page_config(page_title="EPL Predictor", layout="wide")

st.title("⚽ EPL Match Prediction System")


# =================================================
# LOAD AND PREPARE DATA
# =================================================

@st.cache_data
def prepare_data():

    df = load_data(DATA_PATH)

    df = create_features(df)

    # Encode result
    df["FullTimeResult"] = df["FullTimeResult"].map({
        "H": 0,
        "D": 1,
        "A": 2
    })

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

    return df, train_df, test_df, X_train, y_train, X_test, y_test, features


df, train_df, test_df, X_train, y_train, X_test, y_test, features = prepare_data()


# =================================================
# CREATE TABS
# =================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Training",
    "Match Prediction",
    "Season Table",
    "Monte Carlo Simulation"
])


# =================================================
# TAB 1 : MODEL TRAINING
# =================================================

with tab1:

    st.header("Train Model")

    models = get_models()

    model_name = st.selectbox("Select Model", list(models.keys()))

    model = models[model_name]

    if st.button("Train Model"):

        model = train_model(model, X_train, y_train)

        preds, probs = predict_matches(model, X_test)

        accuracy = evaluate_model(y_test, preds)

        st.success(f"Model trained! Accuracy: {accuracy:.3f}")

        st.session_state["model"] = model
        st.session_state["preds"] = preds
        st.session_state["probs"] = probs


# =================================================
# TAB 2 : MATCH PREDICTION
# =================================================

with tab2:

    st.header("Predict Match")

    teams = sorted(pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel()))

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Home Team", teams)

    with col2:
        away_team = st.selectbox("Away Team", teams)

    if home_team == away_team:
        st.warning("Home and Away team cannot be the same.")

    ht_home_goals = st.number_input("Half Time Home Goals", 0, 5)
    ht_away_goals = st.number_input("Half Time Away Goals", 0, 5)

    home_form = st.slider("Home Form (Last 5)", 0.0, 3.0, 1.0)
    away_form = st.slider("Away Form (Last 5)", 0.0, 3.0, 1.0)

    if st.button("Predict Match"):

        if "model" not in st.session_state:
            st.error("Train model first!")
        else:

            X_input = pd.DataFrame([[
                ht_home_goals,
                ht_away_goals,
                home_form,
                away_form
            ]], columns=features)

            pred = st.session_state["model"].predict(X_input)[0]

            probs = st.session_state["model"].predict_proba(X_input)[0]

            result_map = {
                0: "Home Win",
                1: "Draw",
                2: "Away Win"
            }

            st.success(f"Prediction: {result_map[pred]}")

            st.write("### Win Probabilities")

            prob_df = pd.DataFrame({
                "Outcome": ["Home Win", "Draw", "Away Win"],
                "Probability": probs
            })

            st.bar_chart(prob_df.set_index("Outcome"))


# =================================================
# TAB 3 : LEAGUE TABLE
# =================================================

with tab3:

    st.header("Predicted League Table")

    seasons = sorted(df["Season"].unique())

    selected_season = st.selectbox("Select Season", seasons)

    if st.button("Generate League Table"):

        if "preds" not in st.session_state:
            st.error("Train model first!")

        else:

            temp_df = test_df.copy()

            temp_df["PredictedResult"] = st.session_state["preds"]

            season_df = temp_df[temp_df["Season"] == selected_season]

            table = simulate_league(season_df)

            st.write(f"### Predicted Table ({selected_season})")

            st.dataframe(table)

            st.success(f"Predicted Champion: {table.iloc[0]['Team']}")


# =================================================
# TAB 4 : MONTE CARLO SIMULATION
# =================================================

with tab4:

    st.header("Monte Carlo Title Probability")

    seasons = sorted(df["Season"].unique())

    selected_season_mc = st.selectbox("Season for Simulation", seasons)

    n_sim = st.slider("Number of Simulations", 100, 5000, 1000)

    if st.button("Run Simulation"):

        if "probs" not in st.session_state:
            st.error("Train model first!")

        else:

            temp_df = test_df.copy()

            season_df = temp_df[temp_df["Season"] == selected_season_mc]

            season_probs = st.session_state["probs"][season_df.index]

            title_probs, winners_df = simulate_season_mc(
                season_df,
                season_probs,
                n_sim=n_sim
            )

            st.write(f"### Title Probabilities ({selected_season_mc})")

            st.dataframe(title_probs)

            st.bar_chart(
                title_probs.set_index("Team")["TitleProbability"]
            )