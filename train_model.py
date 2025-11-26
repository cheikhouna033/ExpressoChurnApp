import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- Charger le dataset et créer le modèle mini-pipeline ---
df = pd.read_csv("data/Expresso_churn_dataset.csv")
df.replace([" ", "?", "-", "NA", "N/A"], pd.NA, inplace=True)
df = df.drop_duplicates()
df = df.dropna(subset=["CHURN"])
df["CHURN"] = df["CHURN"].astype(int)

features = ["AGE", "REGION", "DATA_VOLUME", "RECHARGE", "TENURE"]
X = df[features]
y = df["CHURN"]

num_cols = ["AGE", "DATA_VOLUME", "RECHARGE", "TENURE"]
cat_cols = ["REGION"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/churn_model_simple.pkl")

# --- Streamlit App ---
st.title("Mini Churn Prediction App (Interactif)")

# Exemples prédéfinis
exemples = {
    "Churn 1": {"AGE": 25, "REGION": "Sénégal", "DATA_VOLUME": 50, "RECHARGE": 500, "TENURE": 10},
    "Fidèle 1": {"AGE": 40, "REGION": "Mauritanie", "DATA_VOLUME": 200, "RECHARGE": 2000, "TENURE": 200},
    "Churn 2": {"AGE": 18, "REGION": "Sénégal", "DATA_VOLUME": 30, "RECHARGE": 0, "TENURE": 5},
    "Fidèle 2": {"AGE": 55, "REGION": "Sénégal", "DATA_VOLUME": 500, "RECHARGE": 5000, "TENURE": 1000},
    "Churn 3": {"AGE": 30, "REGION": "Mauritanie", "DATA_VOLUME": 100, "RECHARGE": 1000, "TENURE": 50},
}

choix = st.selectbox("Choisir un exemple de client :", list(exemples.keys()))
client = exemples[choix]

# Remplir le formulaire automatiquement
AGE = st.number_input("Âge", 10, 100, client["AGE"])
REGION = st.selectbox("Région", df["REGION"].dropna().unique(), index=list(df["REGION"].dropna().unique()).index(client["REGION"]))
DATA_VOLUME = st.number_input("Volume Data (Mo)", 0, 100000, client["DATA_VOLUME"])
RECHARGE = st.number_input("Montant recharge (FCFA)", 0, 200000, client["RECHARGE"])
TENURE = st.number_input("Ancienneté (jours)", 0, 2000, client["TENURE"])

if st.button("Prédire"):
    input_df = pd.DataFrame({
        "AGE": [AGE],
        "REGION": [REGION],
        "DATA_VOLUME": [DATA_VOLUME],
        "RECHARGE": [RECHARGE],
        "TENURE": [TENURE]
    })
    prediction = pipeline.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Le client risque de churn !")
    else:
        st.success("Le client semble fidèle.")
