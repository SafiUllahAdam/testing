import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

st.set_page_config(page_title="Leachate Predictor", layout="wide")

st.title("Leachate Prediction System")
st.write("Rock-based leachate prediction using ML + SHAP explainability")

@st.cache_resource
def load_assets():
    model = joblib.load("rf_model (1).joblib")
    scaler = joblib.load("scaler (1).joblib")
    feature_cols = joblib.load("feature_cols (1).joblib")
    return model, scaler, feature_cols

rf_model, scaler, feature_cols = load_assets()

df_rocks = pd.read_csv("inference_data.csv")

st.sidebar.header("Input Controls")

rock_id = st.sidebar.selectbox(
    "Select Rock",
    sorted(df_rocks["Rock_number"].unique())
)

sequence_len = st.sidebar.slider("Sequence Length (events)", 1, 15, 5)

rock_features = (
    df_rocks[df_rocks["Rock_number"] == rock_id]
    .drop(columns=["Rock_number"])
    .iloc[0]
)

st.subheader("Define Event Sequence")

sequence = []

for i in range(sequence_len):
    with st.expander(f"Event {i+1}", expanded=True):

        event_type = st.selectbox("Event type", ["rain", "snow"], key=f"type_{i}")
        acid = st.slider("Acidity (0 = none, 1 = acidic)", 0.0, 1.0, 0.0, key=f"acid_{i}")
        temp = st.slider("Temperature (°C)", -10.0, 30.0, 5.0, key=f"temp_{i}")
        qty = st.slider("Event quantity", 1.0, 200.0, 150.0, key=f"qty_{i}")

        sequence.append({
            "type": event_type,
            "acid": acid,
            "temp": temp,
            "quantity": qty
        })

def build_event_features(event):

    feats = {}
    feats["is_rain"] = 1 if event["type"] == "rain" else 0
    feats["is_snow"] = 1 if event["type"] == "snow" else 0
    feats["is_acid"] = 1 if event["acid"] > 0 else 0

    feats["acid_rain"] = feats["is_rain"] * feats["is_acid"]
    feats["acid_snow"] = feats["is_snow"] * feats["is_acid"]

    feats["event_quantity"] = event["quantity"]
    feats["Event_quantity"] = event["quantity"]

    feats["Temp"] = event["temp"]
    feats["temperature"] = event["temp"]

    feats["event_intensity"] = event["quantity"] * event["temp"]
    feats["snow_melt_potential"] = feats["is_snow"] * event["temp"]
    feats["acid_intensity"] = feats["is_acid"] * event["quantity"]

    return feats

def explain_event_with_shap_streamlit(x_row, pred, shap_vals, threshold=100, top_k=3):

    feature_names = x_row.index
    contrib = shap_vals
    abs_contrib = np.abs(contrib)
    order = np.argsort(abs_contrib)[::-1]

    risk = "HIGH-RISK" if pred >= threshold else "LOW-RISK"

    def acidity_label(v):
        if v < 0.1:
            return "very low acidity"
        elif v < 0.3:
            return "low acidity"
        elif v < 0.7:
            return "moderate acidity"
        else:
            return "very high acidity"

    def nice_sentence(name, sign):

        qty = x_row.get("event_quantity", x_row.get("Event_quantity", 0))
        acid_val = x_row.get("acid_intensity", 0.0)
        acidity_text = acidity_label(acid_val)

        if name == "event_intensity":
            return (
                f"A strong precipitation event with {acidity_text} pushed the leachate higher."
                if sign > 0 else
                f"A mild precipitation event with {acidity_text} kept the leachate low."
            )

        if name in ["event_quantity", "Event_quantity"]:
            return (
                "Heavy rainfall/snowfall increased the leachate."
                if sign > 0 else
                "Light rainfall/snowfall helped keep the leachate low."
            )

        if name == "Temp":
            return (
                "Warmer temperatures after the event increased the leachate."
                if sign > 0 else
                "Colder temperatures after the event reduced the leachate."
            )

        if name == "acid_intensity":
            return (
                f"The event had {acidity_text}, causing more material to dissolve and increasing the leachate."
                if sign > 0 else
                f"The event had {acidity_text}, so little material dissolved, keeping the leachate low."
            )

        lname = name.lower()

        if lname.startswith("k_"):
            return (
                "Higher potassium levels increased the leachate."
                if sign > 0 else
                "Lower potassium levels helped control the leachate."
            )

        if lname.startswith("carbonate"):
            return (
                "Higher carbonate levels increased the leachate."
                if sign > 0 else
                "Lower carbonate levels helped reduce the leachate."
            )

        return None

    explanations = []

    event_priority = [
        "event_intensity",
        "event_quantity",
        "Event_quantity",
        "Temp",
        "acid_intensity",
        "acid_snow"
    ]

    # PASS 1 — EVENT FEATURES
    for i in order:
        if len(explanations) >= top_k:
            break
        name = feature_names[i]
        if name not in event_priority:
            continue
        sentence = nice_sentence(name, np.sign(contrib[i]))
        if sentence:
            explanations.append(sentence)

    # PASS 2 — CHEMISTRY FEATURES
    for i in order:
        if len(explanations) >= top_k:
            break
        name = feature_names[i]
        lname = name.lower()
        if lname.startswith(("k_", "mg_", "chloride", "carbonate")):
            sentence = nice_sentence(name, np.sign(contrib[i]))
            if sentence:
                explanations.append(sentence)

    if not explanations:
        explanations.append("Overall water chemistry influenced the leachate behaviour.")

    return risk, explanations

if st.button("Run Prediction"):

    explainer = shap.TreeExplainer(rf_model)
    st.subheader("Prediction Results")

    for i, event in enumerate(sequence):

        x = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

        for col in rock_features.index:
            if col in x.columns:
                x.loc[0, col] = rock_features[col]

        event_feats = build_event_features(event)
        for col, val in event_feats.items():
            if col in x.columns:
                x.loc[0, col] = val

        x_scaled = scaler.transform(x)
        pred = rf_model.predict(x_scaled)[0]

        shap_vals = explainer.shap_values(x)[0]

        risk, reasons = explain_event_with_shap_streamlit(x.iloc[0], pred, shap_vals)

        st.markdown("### Explanation")
        st.write(f"**Predicted Leachate:** {pred:.2f}")
        st.write(f"**Risk Level:** {risk}")

        for r in reasons:
            st.write("•", r)
