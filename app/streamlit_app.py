import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import convert_age_to_numeric, diagnosis_to_categories

DATA_PATH = Path("data/processed/diabetic_data_processed.csv")
MODEL_PATH = Path("models/readmission_xgb_pipeline.pkl")
METRICS_PATH = Path("reports/metrics.json")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    feature_df = df.drop(columns=["readmitted_30_days"])

    numeric_stats = feature_df.select_dtypes(exclude="object").median(numeric_only=True)
    mode_series = feature_df.mode(dropna=True).iloc[0]

    baseline = mode_series.copy()
    for column, value in numeric_stats.items():
        baseline[column] = value

    desc_to_id_maps = {
        "admission_type_id": (
            feature_df.groupby("admission_type_id_desc")["admission_type_id"]
            .agg(lambda x: x.mode(dropna=True).iloc[0])
            .to_dict()
        ),
        "discharge_disposition_id": (
            feature_df.groupby("discharge_disposition_id_desc")["discharge_disposition_id"]
            .agg(lambda x: x.mode(dropna=True).iloc[0])
            .to_dict()
        ),
        "admission_source_id": (
            feature_df.groupby("admission_source_id_desc")["admission_source_id"]
            .agg(lambda x: x.mode(dropna=True).iloc[0])
            .to_dict()
        ),
    }

    return feature_df, baseline, desc_to_id_maps


@st.cache_data
def load_metrics() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open() as f:
        return json.load(f)


def get_options(feature_df: pd.DataFrame, column: str) -> list:
    return sorted(
        value for value in feature_df[column].dropna().unique() if str(value).strip()
    )


def compute_comorbidity_index(diag_values) -> int:
    categories = diagnosis_to_categories(diag_values)
    # diagnosis_to_categories uses pandas internally; we do not need CHRONIC_CONDITION_RANGES
    return len(categories)


def main() -> None:
    st.set_page_config(
        page_title="Hospital Readmission Risk",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 30-Day Readmission Risk")
    st.caption(
        "Predict the likelihood of a patient being readmitted within 30 days of discharge."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Trained model not found. Please run `python src/train_model.py` before launching the app."
        )
        st.stop()

    model = load_model()
    feature_df, baseline, desc_to_id_maps = load_dataset()
    metrics = load_metrics()
    default_threshold = float(
        metrics.get("threshold", getattr(model, "threshold_", 0.5))
    )

    with st.sidebar:
        st.header("Model Snapshot")
        if metrics:
            st.metric("Test precision", f"{metrics.get('test_precision', 0):.2f}")
            st.metric("Test recall", f"{metrics.get('test_recall', 0):.2f}")
            st.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.2f}")
            st.caption(
                f"ROC-cutoff currently set to {default_threshold:.2f} "
                f"(scale_pos_weight={metrics.get('scale_pos_weight', 'n/a')})."
            )
        else:
            st.info("Train the model to populate metrics.")

        selected_threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            step=0.01,
            value=default_threshold,
            help="Increase threshold to reduce false positives, decrease to improve recall.",
        )

        cm_path = Path("reports/confusion_matrix.png")
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
    threshold = float(selected_threshold)

    st.subheader("Patient Information")
    feature_options = {
        "race": get_options(feature_df, "race"),
        "gender": get_options(feature_df, "gender"),
        "age": get_options(feature_df, "age"),
        "admission_type_id_desc": get_options(feature_df, "admission_type_id_desc"),
        "discharge_disposition_id_desc": get_options(
            feature_df, "discharge_disposition_id_desc"
        ),
        "admission_source_id_desc": get_options(
            feature_df, "admission_source_id_desc"
        ),
        "max_glu_serum": get_options(feature_df, "max_glu_serum"),
        "A1Cresult": get_options(feature_df, "A1Cresult"),
        "change": get_options(feature_df, "change"),
        "diabetesMed": get_options(feature_df, "diabetesMed"),
        "insulin": get_options(feature_df, "insulin"),
    }

    with st.form("prediction_form"):
        input_data = baseline.copy()

        st.markdown("### Patient Snapshot")
        snapshot_col1, snapshot_col2, snapshot_col3 = st.columns(3)
        input_data["age"] = snapshot_col1.selectbox(
            "Age range",
            feature_options["age"],
            index=feature_options["age"].index(baseline["age"])
            if baseline["age"] in feature_options["age"]
            else 0,
        )
        input_data["gender"] = snapshot_col2.selectbox(
            "Gender",
            feature_options["gender"],
            index=feature_options["gender"].index(baseline["gender"])
            if baseline["gender"] in feature_options["gender"]
            else 0,
        )
        input_data["race"] = snapshot_col3.selectbox(
            "Race",
            feature_options["race"],
            index=feature_options["race"].index(baseline["race"])
            if baseline["race"] in feature_options["race"]
            else 0,
        )

        st.markdown("### Visit Details")
        visit_col1, visit_col2, visit_col3 = st.columns(3)
        input_data["time_in_hospital"] = visit_col1.number_input(
            "Length of stay (days)",
            min_value=1,
            max_value=100,
            value=int(baseline.get("time_in_hospital", 4)),
        )
        input_data["admission_type_id_desc"] = visit_col2.selectbox(
            "Admission type",
            feature_options["admission_type_id_desc"],
            index=feature_options["admission_type_id_desc"].index(
                baseline.get("admission_type_id_desc", feature_options["admission_type_id_desc"][0])
            )
            if feature_options["admission_type_id_desc"]
            else 0,
        )
        input_data["discharge_disposition_id_desc"] = visit_col3.selectbox(
            "Discharge disposition",
            feature_options["discharge_disposition_id_desc"],
            index=feature_options["discharge_disposition_id_desc"].index(
                baseline.get(
                    "discharge_disposition_id_desc",
                    feature_options["discharge_disposition_id_desc"][0],
                )
            )
            if feature_options["discharge_disposition_id_desc"]
            else 0,
        )
        admission_source_col, diagnoses_col = st.columns([2, 1])
        input_data["admission_source_id_desc"] = admission_source_col.selectbox(
            "Admission source",
            feature_options["admission_source_id_desc"],
            index=feature_options["admission_source_id_desc"].index(
                baseline.get(
                    "admission_source_id_desc",
                    feature_options["admission_source_id_desc"][0],
                )
            )
            if feature_options["admission_source_id_desc"]
            else 0,
        )
        input_data["number_diagnoses"] = diagnoses_col.slider(
            "Diagnoses recorded this visit",
            min_value=1,
            max_value=20,
            value=int(baseline.get("number_diagnoses", 8)),
        )

        st.markdown("### Utilization History")
        history_col1, history_col2, history_col3 = st.columns(3)
        input_data["number_outpatient"] = history_col1.slider(
            "Outpatient visits (1y)",
            min_value=0,
            max_value=20,
            value=int(baseline.get("number_outpatient", 0)),
        )
        input_data["number_emergency"] = history_col2.slider(
            "Emergency visits (1y)",
            min_value=0,
            max_value=20,
            value=int(baseline.get("number_emergency", 0)),
        )
        input_data["number_inpatient"] = history_col3.slider(
            "Inpatient stays (1y)",
            min_value=0,
            max_value=20,
            value=int(baseline.get("number_inpatient", 0)),
        )

        st.markdown("### Treatments & Labs")
        labs_col1, labs_col2, labs_col3 = st.columns(3)
        input_data["num_lab_procedures"] = labs_col1.number_input(
            "Lab procedures this stay",
            min_value=0,
            max_value=200,
            value=int(baseline.get("num_lab_procedures", 40)),
        )
        input_data["num_procedures"] = labs_col2.number_input(
            "Clinical procedures",
            min_value=0,
            max_value=20,
            value=int(baseline.get("num_procedures", 1)),
        )
        input_data["num_medications"] = labs_col3.number_input(
            "Medications prescribed",
            min_value=0,
            max_value=50,
            value=int(baseline.get("num_medications", 13)),
        )

        meds_col1, meds_col2, meds_col3 = st.columns(3)
        input_data["change"] = meds_col1.selectbox(
            "Medication changed during stay?",
            feature_options["change"],
            index=feature_options["change"].index(
                baseline.get("change", feature_options["change"][0])
            )
            if feature_options["change"]
            else 0,
        )
        input_data["diabetesMed"] = meds_col2.selectbox(
            "On diabetes medication?",
            feature_options["diabetesMed"],
            index=feature_options["diabetesMed"].index(
                baseline.get("diabetesMed", feature_options["diabetesMed"][0])
            )
            if feature_options["diabetesMed"]
            else 0,
        )
        input_data["insulin"] = meds_col3.selectbox(
            "Insulin regimen",
            feature_options["insulin"],
            index=feature_options["insulin"].index(
                baseline.get("insulin", feature_options["insulin"][0])
            )
            if feature_options["insulin"]
            else 0,
        )

        biomarkers_col1, biomarkers_col2 = st.columns(2)
        input_data["max_glu_serum"] = biomarkers_col1.selectbox(
            "Max glucose serum result",
            feature_options["max_glu_serum"],
            index=feature_options["max_glu_serum"].index(
                baseline.get("max_glu_serum", feature_options["max_glu_serum"][0])
            )
            if feature_options["max_glu_serum"]
            else 0,
        )
        input_data["A1Cresult"] = biomarkers_col2.selectbox(
            "Most recent A1C",
            feature_options["A1Cresult"],
            index=feature_options["A1Cresult"].index(
                baseline.get("A1Cresult", feature_options["A1Cresult"][0])
            )
            if feature_options["A1Cresult"]
            else 0,
        )

        with st.expander("Diagnosis codes (optional)", expanded=False):
            st.caption(
                "Provide ICD-9 codes to refine the comorbidity profile. "
                "Defaults fall back to the dataset baseline."
            )
            diag_col1, diag_col2, diag_col3 = st.columns(3)
            input_data["diag_1"] = diag_col1.text_input(
                "Primary diagnosis", value=str(baseline.get("diag_1", "250.00"))
            )
            input_data["diag_2"] = diag_col2.text_input(
                "Secondary diagnosis", value=str(baseline.get("diag_2", "401.9"))
            )
            input_data["diag_3"] = diag_col3.text_input(
                "Additional diagnosis", value=str(baseline.get("diag_3", "414.01"))
            )

        submitted = st.form_submit_button("Predict Readmission Risk", type="primary")

    if not submitted:
        st.info("Fill in the patient information and press **Predict Readmission Risk**.")
        st.stop()

    input_data["previous_admissions"] = (
        input_data["number_inpatient"]
        + input_data["number_outpatient"]
        + input_data["number_emergency"]
    )

    input_data["age_num"] = convert_age_to_numeric(input_data["age"])
    input_data["comorbidity_index"] = compute_comorbidity_index(
        [input_data["diag_1"], input_data["diag_2"], input_data["diag_3"]]
    )

    # Ensure original ID columns are aligned with descriptions when available
    input_data["admission_type_id"] = desc_to_id_maps["admission_type_id"].get(
        input_data["admission_type_id_desc"], baseline.get("admission_type_id", 0)
    )
    input_data["discharge_disposition_id"] = desc_to_id_maps["discharge_disposition_id"].get(
        input_data["discharge_disposition_id_desc"],
        baseline.get("discharge_disposition_id", 0),
    )
    input_data["admission_source_id"] = desc_to_id_maps["admission_source_id"].get(
        input_data["admission_source_id_desc"], baseline.get("admission_source_id", 0)
    )

    input_dataframe = pd.DataFrame([input_data])
    probability = model.predict_proba(input_dataframe)[0, 1]
    prediction = int(probability >= threshold)

    st.success(
        f"Predicted risk of readmission within 30 days: **{probability:.1%}** "
        f"({'High' if prediction == 1 else 'Low'} risk at threshold {threshold:.2f})."
    )

    st.subheader("Key Inputs")
    st.dataframe(input_dataframe, use_container_width=True)


if __name__ == "__main__":
    main()

