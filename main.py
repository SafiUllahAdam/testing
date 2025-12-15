import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# -------------------------------
# Load artifacts
# -------------------------------
rf_model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_cols = joblib.load("feature_cols.joblib")
df = pd.read_csv("inference_data.csv")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Leachate Prediction", layout="centered")
st.title("Leachate Prediction System")

# -------------------------------
# User selection
# -------------------------------
rock = st.selectbox("Select Rock", sorted(df["Rock_number"].unique()))
subset = df[df["Rock_number"] == rock]

timestep = st.selectbox(
    "Select Timestep (1–15)",
    sorted(subset["Timestep"].unique())
)

row = subset[subset["Timestep"] == timestep]

if row.empty:
    st.error("No data for this selection.")
    st.stop()

# -------------------------------
# Show event info
# -------------------------------
st.subheader("Event Details")
st.write(f"• Event quantity: {row['Event_quantity'].values[0]}")
st.write(f"• Temperature: {row['Temp'].values[0]}")
st.write(f"• Acidic: {'Yes' if row['is_acid'].values[0]==1 else 'No'}")

# -------------------------------
# Prediction
# -------------------------------
X = row[feature_cols]
X_scaled = scaler.transform(X)

prediction = rf_model.predict(X_scaled)[0]
risk = "HIGH RISK" if prediction >= 100 else "LOW RISK"

st.subheader("Prediction")
st.write(f"**Predicted Leachate:** {prediction:.2f}")
st.write(f"**Risk Level:** {risk}")

# -------------------------------
# SHAP explanation
# -------------------------------
st.subheader("Why this prediction?")

explainer = shap.TreeExplainer(rf_model)
shap_vals = explainer.shap_values(X_scaled)[0]

shap_df = pd.DataFrame({
    "Feature": feature_cols,
    "SHAP": shap_vals
})

shap_df = shap_df.reindex(
    shap_df.SHAP.abs().sort_values(ascending=False).index
)

def explain_feature(name, value):
    if value > 0:
        return f"{name} increased leachate"
    else:
        return f"{name} helped reduce leachate"

for _, r in shap_df.head(3).iterrows():
    st.write("•", explain_feature(r["Feature"], r["SHAP"]))