
import streamlit as st
import pandas as pd
import sqlite3
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Crop-Growing Condition Monitor",
    page_icon="🌱",
    layout="wide"
)

# Project files
PROJECT_FOLDER = Path.home() / "Desktop" / "MLNG"
MODEL_PATH = PROJECT_FOLDER / "crop_suitability_model.pkl"
DATABASE_PATH = PROJECT_FOLDER / "crop_growing_database.db"

st.title("🌱 Crop-Growing Condition Monitor")
st.write(
    "Monitor growing conditions using a trained machine-learning model "
    "and review historical sensor records."
)

# Load the trained model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the crop suitability model: {e}")
    st.stop()

# Sensor inputs
st.sidebar.header("Sensor Measurements")

temperature = st.sidebar.number_input(
    "Temperature (°C)", value=25.0
)
humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)
soil_moisture = st.sidebar.number_input(
    "Soil Moisture (%)",
    min_value=0.0,
    max_value=100.0,
    value=45.0
)
light = st.sidebar.number_input(
    "Light Intensity",
    min_value=0.0,
    value=500.0
)

if st.sidebar.button("Check Conditions"):
    # Keep the same feature names and order used during model training.
    sensor_readings = pd.DataFrame([{
        "Temperature_C": temperature,
        "Humidity_pct": humidity,
        "Soil_Moisture_pct": soil_moisture,
        "Light_Intensity": light
    }])

    try:
        prediction = model.predict(sensor_readings)[0]

        st.subheader("Crop Suitability Prediction")

        if prediction == "Suitable Conditions":
            st.success("🌱 Suitable Conditions")
        else:
            st.error("⚠️ Unsuitable Conditions")

        # Show the current sensor readings
        st.subheader("Current Sensor Readings")

        readings = pd.DataFrame({
            "Measurement": [
                "Temperature (°C)",
                "Humidity (%)",
                "Soil Moisture (%)",
                "Light Intensity"
            ],
            "Value": [
                temperature,
                humidity,
                soil_moisture,
                light
            ]
        })

        st.dataframe(readings, width="stretch")

        # Show model confidence if the model supports probabilities.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sensor_readings)[0]
            classes = model.classes_
            confidence = probabilities[list(classes).index(prediction)]

            st.metric(
                "Model Confidence",
                f"{confidence:.1%}"
            )

    except Exception as e:
        st.error(f"Could not classify these sensor readings: {e}")

st.divider()
st.subheader("Historical Dataset Summary")

if DATABASE_PATH.exists():
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            historical_df = pd.read_sql_query(
                "SELECT * FROM crop_growing_data", conn
            )

        if historical_df.empty:
            st.info("The historical database currently has no records.")

        elif "Condition_Status" in historical_df.columns:
            counts = historical_df["Condition_Status"].value_counts()

            col1, col2 = st.columns(2)

            col1.metric(
                "Suitable Records",
                int(counts.get("Suitable Conditions", 0))
            )
            col2.metric(
                "Unsuitable Records",
                int(counts.get("Unsuitable Conditions", 0))
            )

            st.bar_chart(counts)

            st.subheader("Recent Dataset Records")
            st.dataframe(
                historical_df.tail(10),
                width="stretch"
            )

        else:
            st.warning(
                "The database does not yet contain Condition_Status. "
                "Run the condition-classification notebook cell first."
            )

    except Exception as e:
        st.error(f"Could not read the database: {e}")

else:
    st.warning(
        "Database file not found. Save your dataset to SQLite first."
    )

st.caption(
    "The model was trained using labels generated from illustrative "
    "thresholds. Its predictions imitate those labels and do not, by "
    "themselves, prove that conditions are suitable for a particular crop."
)
