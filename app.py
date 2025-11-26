import streamlit as st
import pandas as pd
import joblib

st.title("Expresso Churn Prediction App")
st.write("Prédisez si un client risque de se désabonner.")

# -----------------------------------------------------
# 🔁 Charger le modèle compressé depuis le dépôt GitHub
# -----------------------------------------------------
@st.cache_resource
def load_model():
    # Chemin relatif dans le repo
    pipeline = joblib.load("churn_model.pkl")
    return pipeline

pipeline = load_model()

# Colonnes du preprocessing
all_cols = pipeline.named_steps['preprocess'].transformers_[0][2] + \
           pipeline.named_steps['preprocess'].transformers_[1][2]

# -----------------------------------------------------
# 🔧 Interface utilisateur
# -----------------------------------------------------
AGE = st.number_input("Âge", 10, 100, 30)
REGION = st.selectbox("Région", ["Sénégal", "Mauritanie"])
DATA_VOLUME = st.number_input("Volume Data (Mo)", 0, 100000, 500)
RECHARGE = st.number_input("Montant recharge (FCFA)", 0, 200000, 5000)
TENURE = st.number_input("Ancienneté (jours)", 0, 2000, 100)

# -----------------------------------------------------
# 🔮 Prédiction
# -----------------------------------------------------
if st.button("Prédire"):
    input_df = pd.DataFrame(columns=all_cols)

    for col in all_cols:
        input_df.loc[0, col] = 0 if col in pipeline.named_steps['preprocess'].transformers_[0][2] else ""

    input_df.loc[0, "AGE"] = AGE
    input_df.loc[0, "REGION"] = REGION
    input_df.loc[0, "DATA_VOLUME"] = DATA_VOLUME
    input_df.loc[0, "RECHARGE"] = RECHARGE
    input_df.loc[0, "TENURE"] = TENURE

    prediction = pipeline.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Le client risque de churn !")
    else:
        st.success("✔️ Le client semble fidèle.")
