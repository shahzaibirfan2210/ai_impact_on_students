from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "burnout_model.joblib"

st.set_page_config(
    page_title="Student Burnout Risk Estimator",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


bundle = load_model()
st.title("Student Burnout Risk Estimator")
st.write(
    "Enter the student indicators below to estimate the model's burnout-risk "
    "category. Results are statistical estimates, not a diagnosis."
)
st.caption(
    f"Model: {bundle['selected_model']} · "
    f"Held-out test accuracy: {bundle['test_metrics']['accuracy']:.1%}"
)

with st.form("prediction_form"):
    values = {}
    left, right = st.columns(2)
    for index, field in enumerate(bundle["input_schema"]):
        column = left if index % 2 == 0 else right
        with column:
            if field["kind"] == "category":
                choices = field["choices"]
                default_index = (
                    choices.index(field["default"])
                    if field["default"] in choices
                    else 0
                )
                values[field["name"]] = st.selectbox(
                    field["name"].replace("_", " "),
                    choices,
                    index=default_index,
                )
            elif field["kind"] == "boolean":
                values[field["name"]] = st.checkbox(
                    field["name"].replace("_", " "),
                    value=field["default"],
                )
            else:
                minimum = field["minimum"]
                maximum = field["maximum"]
                default = min(max(field["default"], minimum), maximum)
                if field["integer"]:
                    values[field["name"]] = st.number_input(
                        field["name"].replace("_", " "),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(round(default)),
                        step=1,
                    )
                else:
                    values[field["name"]] = st.number_input(
                        field["name"].replace("_", " "),
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        step=0.1,
                        format="%.2f",
                    )
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    sample = pd.DataFrame(
        [{column: values[column] for column in bundle["feature_columns"]}]
    )
    pipeline = bundle["pipeline"]
    prediction = pipeline.predict(sample)[0]
    st.subheader(f"Estimated category: {prediction}")
    if hasattr(pipeline, "predict_proba"):
        probabilities = pd.Series(
            pipeline.predict_proba(sample)[0],
            index=pipeline.classes_,
            name="Estimated probability",
        ).sort_values(ascending=False)
        st.write("Model class probabilities")
        st.bar_chart(probabilities)
    st.info(
        "This model estimates patterns in its training dataset. It has not been "
        "validated for clinical, counseling, or high-stakes decisions."
    )

with st.expander("About this model"):
    st.write(
        f"The selected model was chosen using a separate validation split. "
        f"On a held-out test set of {bundle['test_metrics']['test_rows']:,} rows, "
        f"its accuracy was {bundle['test_metrics']['accuracy']:.1%}, balanced "
        f"accuracy was {bundle['test_metrics']['balanced_accuracy']:.1%}, and "
        f"macro F1 was {bundle['test_metrics']['macro_f1']:.3f}."
    )
    st.write(
        "Performance can be limited by the source data. Review class-level "
        "precision and recall before using predictions to inform decisions."
    )
