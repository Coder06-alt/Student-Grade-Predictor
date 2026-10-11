"""Streamlit page for the Student Grade Predictor.

Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st

import grade_model as gm

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Training model...")
def get_bundle(use_previous_grades: bool):
    """Train once per setting and reuse across reruns."""
    return gm.train(use_previous_grades)


st.title("🎓 Student Grade Predictor")
st.caption("Random Forest regression predicting the final grade G3 (0-20) from academic and student-related features.")

use_prev = st.sidebar.toggle("Use previous grades (G1, G2)", value=True)
st.sidebar.markdown(
    "**On:** very informative, but only usable once earlier grades exist.\n\n"
    "**Off:** harder 'early warning' mode using background features only."
)
bundle = get_bundle(use_prev)

tab_predict, tab_model = st.tabs(["Predict", "Model info"])

with tab_predict:
    st.subheader("Enter student details")
    c1, c2 = st.columns(2)

    inputs = {}
    if use_prev:
        inputs["G1"] = c1.slider("First period grade (G1)", 0, 20, 10)
        inputs["G2"] = c2.slider("Second period grade (G2)", 0, 20, 10)
    inputs["studytime"] = c1.select_slider(
        "Weekly study time", options=[1, 2, 3, 4], value=2,
        format_func=lambda v: {1: "<2 h", 2: "2-5 h", 3: "5-10 h", 4: ">10 h"}[v])
    inputs["failures"] = c2.select_slider("Past class failures", options=[0, 1, 2, 3], value=0)
    inputs["absences"] = c1.slider("Absences", 0, 75, 4)
    inputs["age"] = c2.slider("Age", 15, 22, 17)
    inputs["Medu"] = c1.slider("Mother's education (0-4)", 0, 4, 2)
    inputs["Fedu"] = c2.slider("Father's education (0-4)", 0, 4, 2)
    inputs["Dalc"] = c1.slider("Weekday alcohol (1-5)", 1, 5, 1)
    inputs["Walc"] = c2.slider("Weekend alcohol (1-5)", 1, 5, 2)
    inputs["higher"] = c1.selectbox("Wants higher education?", ["yes", "no"])
    inputs["internet"] = c2.selectbox("Internet at home?", ["yes", "no"])

    st.caption("Fields not shown here use typical values from the training data.")

    if st.button("Predict final grade", type="primary"):
        grade = gm.predict_grade(bundle, inputs)
        st.metric("Predicted final grade (G3)", f"{grade:.1f} / 20")
        st.progress(grade / 20)
        rmse = bundle["rmse"]
        st.info(
            f"Typical error of this model on unseen students is about ±{rmse:.1f} grade points, "
            f"so a reasonable range is {max(grade - rmse, 0):.1f} to {min(grade + rmse, 20):.1f}."
        )
        if grade < 10:
            st.warning("Predicted below 10: this student may need support.")

with tab_model:
    st.subheader("How good is the model?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test R²", f"{bundle['r2']:.2f}")
    m2.metric("Test RMSE", f"{bundle['rmse']:.2f}")
    m3.metric("Training rows", bundle["n_rows"])
    st.caption("Measured on a held-out 20% test set. R² = share of variance explained; RMSE is in grade points.")

    st.subheader("Top features")
    top = bundle["importances"].head(10)
    st.bar_chart(pd.DataFrame({"importance": top}))
    st.caption("Random Forest feature importances. Categorical features appear as one-hot columns (e.g. Mjob_teacher).")
