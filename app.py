import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="IT5006 Predictive Policing Demo",
    page_icon="📊",
    layout="wide"
)

st.title("📊 IT5006 Predictive Policing Demo")
st.caption("Crime tier risk prediction using trained models, temporal features, and lag features.")


# =========================
# File paths
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "crime_independent_models.joblib"
SCALER_FILE = BASE_DIR / "data_scaler.joblib"
FEATURE_FILE = BASE_DIR / "feature_names.joblib"

PERFORMANCE_FILE = BASE_DIR / "performance_df.csv"
CV_FILE = BASE_DIR / "cv_results_df.csv"
SPATIAL_TEMPORAL_FILE = BASE_DIR / "spatial_temporal_df.csv"


# =========================
# Load resources
# =========================
@st.cache_resource
def load_artifacts():
    models = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_FILE)
    return models, scaler, feature_names


@st.cache_data
def load_tables():
    performance_df = pd.read_csv(PERFORMANCE_FILE)
    cv_results_df = pd.read_csv(CV_FILE)
    spatial_temporal_df = pd.read_csv(SPATIAL_TEMPORAL_FILE)
    return performance_df, cv_results_df, spatial_temporal_df


def check_required_files():
    required = [
        MODEL_FILE,
        SCALER_FILE,
        FEATURE_FILE,
        PERFORMANCE_FILE,
        CV_FILE,
        SPATIAL_TEMPORAL_FILE,
    ]
    missing = [str(f.name) for f in required if not f.exists()]
    return missing


missing_files = check_required_files()
if missing_files:
    st.error("Missing required files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

try:
    models, scaler, feature_names = load_artifacts()
    performance_df, cv_results_df, spatial_temporal_df = load_tables()
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()


# =========================
# Sidebar
# =========================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Model Performance",
        "Make Prediction",
        "How to Use"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Loaded files")
st.sidebar.write("✅ crime_independent_models.joblib")
st.sidebar.write("✅ data_scaler.joblib")
st.sidebar.write("✅ feature_names.joblib")
st.sidebar.write("✅ performance_df.csv")
st.sidebar.write("✅ cv_results_df.csv")
st.sidebar.write("✅ spatial_temporal_df.csv")


# =========================
# User-friendly labels
# =========================
FEATURE_LABELS = {
    "month_sin": "Month cycle (sin)",
    "month_cos": "Month cycle (cos)",
    "dow_sin": "Day-of-week cycle (sin)",
    "dow_cos": "Day-of-week cycle (cos)",
    "tier1_lag_7d": "Tier 1 incidents in the past 7 days",
    "tier1_lag_30d": "Tier 1 incidents in the past 30 days",
    "tier2_lag_7d": "Tier 2 incidents in the past 7 days",
    "tier2_lag_30d": "Tier 2 incidents in the past 30 days",
    "tier3_lag_7d": "Tier 3 incidents in the past 7 days",
    "tier3_lag_30d": "Tier 3 incidents in the past 30 days",
    "tier4_lag_7d": "Tier 4 incidents in the past 7 days",
    "tier4_lag_30d": "Tier 4 incidents in the past 30 days",
}


# =========================
# Helper functions
# =========================
def prepare_input_dataframe(input_dict, ordered_features):
    row = {feature: float(input_dict.get(feature, 0.0)) for feature in ordered_features}
    return pd.DataFrame([row], columns=ordered_features)


def predict_all_tiers(model_name, input_df):
    """
    models[model_name] is a list of 4 binary classifiers:
    [tier1_model, tier2_model, tier3_model, tier4_model]
    """
    transformed = scaler.transform(input_df)
    model_list = models[model_name]

    results = []
    for idx, clf in enumerate(model_list, start=1):
        prob = float(clf.predict_proba(transformed)[0, 1])
        pred = int(prob >= 0.5)
        results.append({
            "Crime Tier": f"Tier {idx}",
            "Predicted Label (0/1)": pred,
            "Predicted Probability": round(prob, 4)
        })

    result_df = pd.DataFrame(results).sort_values(
        by="Predicted Probability", ascending=False
    ).reset_index(drop=True)

    return result_df


def get_best_auc_rows(perf_df):
    best_rows = perf_df.loc[perf_df.groupby("Crime Tier")["AUC-ROC"].idxmax()].copy()
    best_rows = best_rows.sort_values("Crime Tier").reset_index(drop=True)
    return best_rows


# =========================
# Page: Overview
# =========================
if page == "Overview":
    st.subheader("Project Overview")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(
            """
This demo presents a **proof-of-concept predictive policing application** built from trained machine learning artifacts.

The current app supports:
- loading trained models
- comparing model performance
- entering temporal and lag-based features
- generating **crime tier risk probabilities**

The available models are:
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network
            """
        )

    with col2:
        friendly_feature_lines = "\n".join(
            [f"- {FEATURE_LABELS.get(f, f)}" for f in feature_names]
        )
        st.info(
            f"""
    **Model input features ({len(feature_names)} total):**

    {friendly_feature_lines}
            """
        )

    st.markdown("---")
    st.subheader("Best AUC-ROC by Crime Tier")
    best_auc_df = get_best_auc_rows(performance_df)
    st.dataframe(best_auc_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Artifacts Summary")

    summary_df = pd.DataFrame({
        "Artifact": [
            "Trained models",
            "Scaler",
            "Feature names",
            "Performance table",
            "Cross-validation table",
            "Spatial-temporal table",
        ],
        "Description": [
            "4 model families, each with 4 binary classifiers for crime tiers",
            "StandardScaler used before prediction",
            "Ordered list of 12 required input features",
            "Precision / Recall / F1-Score / AUC-ROC",
            "Cross-validation ROC-AUC summary",
            "Weekday / Weekend accuracy and PAI",
        ]
    })
    st.dataframe(summary_df, use_container_width=True)


# =========================
# Page: Model Performance
# =========================
elif page == "Model Performance":
    st.subheader("Model Evaluation Results")

    tab1, tab2, tab3 = st.tabs([
        "Performance Metrics",
        "Cross-Validation",
        "Spatial & Temporal"
    ])

    with tab1:
        st.markdown("### Standard Classification Metrics")
        st.dataframe(performance_df, use_container_width=True)

        st.markdown("### Filter by Model")
        model_options = ["All"] + sorted(performance_df["Model"].unique().tolist())
        selected_model = st.selectbox("Select model", model_options, key="perf_model")
        if selected_model != "All":
            filtered_perf = performance_df[performance_df["Model"] == selected_model]
            st.dataframe(filtered_perf, use_container_width=True)

    with tab2:
        st.markdown("### Cross-Validation Results")
        st.dataframe(cv_results_df, use_container_width=True)

    with tab3:
        st.markdown("### Spatial / Temporal Robustness")
        st.dataframe(spatial_temporal_df, use_container_width=True)


# =========================
# Page: Make Prediction
# =========================
elif page == "Make Prediction":
    st.subheader("Make a Prediction")

    st.markdown(
        """
Enter values for the 12 required features below.  
Then select a trained model and click **Run Prediction**.
        """
    )

    model_names = list(models.keys())
    selected_model_name = st.selectbox("Choose model", model_names)

    st.markdown("### Input Features")

    # Default values
    default_values = {
        "month_sin": 0.0,
        "month_cos": 1.0,
        "dow_sin": 0.0,
        "dow_cos": 1.0,
        "tier1_lag_7d": 0,
        "tier1_lag_30d": 0,
        "tier2_lag_7d": 0,
        "tier2_lag_30d": 0,
        "tier3_lag_7d": 0,
        "tier3_lag_30d": 0,
        "tier4_lag_7d": 0,
        "tier4_lag_30d": 0,
    }

    input_values = {}

    col1, col2 = st.columns(2)

    with col1:
        input_values["month_sin"] = st.number_input(
            FEATURE_LABELS["month_sin"],
            value=float(default_values["month_sin"]),
            format="%.4f"
        )
        input_values["month_cos"] = st.number_input(
            FEATURE_LABELS["month_cos"],
            value=float(default_values["month_cos"]),
            format="%.4f"
        )
        input_values["dow_sin"] = st.number_input(
            FEATURE_LABELS["dow_sin"],
            value=float(default_values["dow_sin"]),
            format="%.4f"
        )
        input_values["dow_cos"] = st.number_input(
            FEATURE_LABELS["dow_cos"],
            value=float(default_values["dow_cos"]),
            format="%.4f"
        )
        input_values["tier1_lag_7d"] = st.number_input(
            FEATURE_LABELS["tier1_lag_7d"],
            min_value=0,
            value=int(default_values["tier1_lag_7d"]),
            step=1
        )
        input_values["tier1_lag_30d"] = st.number_input(
            FEATURE_LABELS["tier1_lag_30d"],
            min_value=0,
            value=int(default_values["tier1_lag_30d"]),
            step=1
        )

    with col2:
        input_values["tier2_lag_7d"] = st.number_input(
            FEATURE_LABELS["tier2_lag_7d"],
            min_value=0,
            value=int(default_values["tier2_lag_7d"]),
            step=1
        )
        input_values["tier2_lag_30d"] = st.number_input(
            FEATURE_LABELS["tier2_lag_30d"],
            min_value=0,
            value=int(default_values["tier2_lag_30d"]),
            step=1
        )
        input_values["tier3_lag_7d"] = st.number_input(
            FEATURE_LABELS["tier3_lag_7d"],
            min_value=0,
            value=int(default_values["tier3_lag_7d"]),
            step=1
        )
        input_values["tier3_lag_30d"] = st.number_input(
            FEATURE_LABELS["tier3_lag_30d"],
            min_value=0,
            value=int(default_values["tier3_lag_30d"]),
            step=1
        )
        input_values["tier4_lag_7d"] = st.number_input(
            FEATURE_LABELS["tier4_lag_7d"],
            min_value=0,
            value=int(default_values["tier4_lag_7d"]),
            step=1
        )
        input_values["tier4_lag_30d"] = st.number_input(
            FEATURE_LABELS["tier4_lag_30d"],
            min_value=0,
            value=int(default_values["tier4_lag_30d"]),
            step=1
        )

    st.markdown("### Quick Example")
    if st.button("Load example values"):
        example_values = {
            "month_sin": 0.5,
            "month_cos": 0.8660,
            "dow_sin": 0.7818,
            "dow_cos": 0.6235,
            "tier1_lag_7d": 8,
            "tier1_lag_30d": 30,
            "tier2_lag_7d": 14,
            "tier2_lag_30d": 55,
            "tier3_lag_7d": 20,
            "tier3_lag_30d": 80,
            "tier4_lag_7d": 10,
            "tier4_lag_30d": 40,
        }
        st.session_state["example_values"] = example_values
        st.success("Example loaded. You can manually copy these values into the input boxes.")

    if "example_values" in st.session_state:
        st.json(st.session_state["example_values"])

    if st.button("Run Prediction", type="primary"):
        try:
            input_df = prepare_input_dataframe(input_values, feature_names)
            prediction_df = predict_all_tiers(selected_model_name, input_df)

            st.success("Prediction completed.")
            st.markdown("### Prediction Result")
            st.dataframe(prediction_df, use_container_width=True)

            top_tier = prediction_df.iloc[0]
            st.info(
                f"Highest predicted risk: **{top_tier['Crime Tier']}** "
                f"with probability **{top_tier['Predicted Probability']:.4f}** "
                f"using **{selected_model_name}**."
            )

            st.markdown("### Input Used")
            display_input_df = input_df.rename(columns=FEATURE_LABELS)
            st.dataframe(display_input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# =========================
# Page: How to Use
# =========================
elif page == "How to Use":
    st.subheader("How to Use This Demo")

    st.markdown(
        """
### Steps
1. Open the **Make Prediction** page.
2. Choose one of the trained models.
3. Enter values for the 12 required features.
4. Click **Run Prediction**.
5. Review the predicted probabilities for **Tier 1 to Tier 4**.

### Notes
- This is a **proof-of-concept deployment**.
- Inputs are based on:
  - cyclic temporal features (`month_sin`, `month_cos`, `dow_sin`, `dow_cos`)
  - lag-based recent crime counts for each tier
- The app applies the saved **StandardScaler** before prediction.
- Each model returns probabilities for 4 separate binary tier classifiers.

### Interpretation
- A higher probability means the selected model estimates a higher likelihood for that tier.
- Since each tier is modeled separately, probabilities across tiers do **not** have to sum to 1.

### Recommended Demo Flow
- Show the **Overview** page first.
- Then show **Model Performance**.
- Finally go to **Make Prediction** and run one example live.
        """
    )

    st.markdown("---")
    st.markdown("### Input Features Used by the Model")
    friendly_feature_list = [FEATURE_LABELS.get(f, f) for f in feature_names]
    st.write(friendly_feature_list)