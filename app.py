import streamlit as st
import joblib
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease CDSS",
    page_icon="🫀",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("models/heart_disease_xgboost.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🫀 Heart Disease Clinical Decision Support System")

st.markdown(
    """
    This system uses a machine learning model to estimate
    heart disease risk based on selected patient information.
    """
)

st.info(
    "⚠️ Educational prototype only. "
    "This system does not provide a medical diagnosis."
)

# --------------------------------------------------
# PATIENT INFORMATION
# --------------------------------------------------

st.header("Patient Information")

col1, col2, col3 = st.columns(3)

# --------------------------------------------------
# COLUMN 1 — DEMOGRAPHICS
# --------------------------------------------------

with col1:

    st.subheader("👤 Demographics")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
            "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        help="Encoded chest pain category from the dataset."
    )

# --------------------------------------------------
# COLUMN 2 — VITAL SIGNS
# --------------------------------------------------

with col2:

    st.subheader("❤️ Vital Signs")

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=130
    )

    chol = st.number_input(
        "Serum Cholesterol",
        min_value=50,
        max_value=700,
        value=240
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

# --------------------------------------------------
# COLUMN 3 — CLINICAL FEATURES
# --------------------------------------------------

with col3:

    st.subheader("🩺 Clinical Features")

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2]
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

# --------------------------------------------------
# ADDITIONAL FEATURES
# --------------------------------------------------

st.subheader("Additional Clinical Features")

col4, col5, col6 = st.columns(3)

with col4:

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2]
    )

with col5:

    ca = st.selectbox(
        "Number of Major Vessels",
        options=[0, 1, 2, 3]
    )

with col6:

    thal = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3]
    )

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

st.divider()

analyze = st.button(
    "🔍 Analyze Heart Disease Risk",
    use_container_width=True
)

if analyze:

    # Create patient dataframe
    patient_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=feature_names
    )

    # Model prediction
    prediction = model.predict(patient_data)[0]

    probability = model.predict_proba(
        patient_data
    )[0][1]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.divider()

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Model-Estimated Risk Score",
            f"{probability * 100:.1f}%"
        )

    with result_col2:

        if prediction == 1:
            st.metric(
                "Classification",
                "Higher likelihood"
            )
        else:
            st.metric(
                "Classification",
                "Lower likelihood"
            )

    # --------------------------------------------------
    # RESULT MESSAGE
    # --------------------------------------------------

    if prediction == 1:

        st.error(
            "The model predicts a higher likelihood "
            "of heart disease."
        )

    else:

        st.success(
            "The model predicts a lower likelihood "
            "of heart disease."
        )

    # --------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------

    st.warning(
        "This result is generated by a machine learning "
        "model for educational purposes. It is not a "
        "medical diagnosis and should not replace "
        "professional clinical evaluation."
    )

    # --------------------------------------------------
    # PATIENT DATA SUMMARY
    # --------------------------------------------------

    with st.expander("View Patient Data Used by Model"):

        st.dataframe(
            patient_data,
            use_container_width=True
        )
        # --------------------------------------------------
# CREATOR WATERMARK
# --------------------------------------------------

st.markdown(
    """
    <style>
    .creator-watermark {
        text-align: center;
        padding: 20px 0 10px 0;
        margin-top: 40px;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        color: rgba(255, 255, 255, 0.55);
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    </style>

    <div class="creator-watermark">
        Created by - <b>Sk Mohammad Zunaid</b>
        &nbsp; | &nbsp;
        Reg No - <b>25BHI10085</b>
    </div>
    """,
    unsafe_allow_html=True
)