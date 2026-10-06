import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreditScore AI",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "credit_scoring_model.pkl"
)

METADATA_PATH = (
    BASE_DIR / "models" / "model_metadata.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


model = load_model()
metadata = load_metadata()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 100%
        );
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.7rem;
        font-weight: 750;
        margin: 0 0 0.5rem 0;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.82;
        margin: 0;
    }

    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        background: #0f172a;
        margin-bottom: 1rem;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("CreditScore AI")

    st.caption(
        "Machine Learning Credit Risk Assessment"
    )

    st.divider()

    st.subheader("Model")

    st.write("Random Forest Classifier")

    st.metric(
        "CV ROC-AUC",
        f"{metadata['cv_roc_auc']:.3f}"
    )

    st.metric(
        "Test ROC-AUC",
        f"{metadata['test_roc_auc']:.3f}"
    )

    st.divider()

    st.subheader("Project")

    st.write(
        "Creditworthiness prediction using "
        "machine learning."
    )

    st.divider()

    st.caption(
        "Educational ML project. Predictions should "
        "not be treated as financial advice."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>CreditScore AI</h1>
        <p>
            AI-powered credit risk assessment
            using machine learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "Enter the applicant's financial information "
    "to estimate their credit risk."
)


# ============================================================
# CODE → HUMAN READABLE OPTIONS
# ============================================================

checking_account = {
    "A11": "Negative balance",
    "A12": "0 – 200 DM",
    "A13": "200+ DM / salary assignment ≥ 1 year",
    "A14": "No checking account"
}

credit_history = {
    "A30": "No credits / all credits paid duly",
    "A31": "All credits at this bank paid duly",
    "A32": "Existing credits paid duly",
    "A33": "Previous payment delays",
    "A34": "Critical account / other existing credits"
}

purpose = {
    "A40": "New car",
    "A41": "Used car",
    "A42": "Furniture / equipment",
    "A43": "Radio / television",
    "A44": "Domestic appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A48": "Retraining",
    "A49": "Business",
    "A410": "Other"
}

savings = {
    "A61": "Less than 100 DM",
    "A62": "100 – 500 DM",
    "A63": "500 – 1000 DM",
    "A64": "1000+ DM",
    "A65": "Unknown / no savings account"
}

employment = {
    "A71": "Unemployed",
    "A72": "Less than 1 year",
    "A73": "1 – 4 years",
    "A74": "4 – 7 years",
    "A75": "7+ years"
}

personal_status = {
    "A91": "Male – divorced / separated",
    "A92": "Female – divorced / separated / married",
    "A93": "Male – single",
    "A94": "Male – married / widowed",
    "A95": "Female – single"
}

debtors = {
    "A101": "None",
    "A102": "Co-applicant",
    "A103": "Guarantor"
}

property_options = {
    "A121": "Real estate",
    "A122": "Savings agreement / life insurance",
    "A123": "Car or other property",
    "A124": "Unknown / no property"
}

installment_plans = {
    "A141": "Bank",
    "A142": "Stores",
    "A143": "None"
}

housing = {
    "A151": "Rent",
    "A152": "Own",
    "A153": "For free"
}

job = {
    "A171": "Unemployed / unskilled non-resident",
    "A172": "Unskilled resident",
    "A173": "Skilled employee / official",
    "A174": "Management / self-employed / highly qualified"
}

telephone = {
    "A191": "No telephone",
    "A192": "Registered telephone"
}

foreign_worker = {
    "A201": "Yes",
    "A202": "No"
}


# ============================================================
# APPLICATION FORM
# ============================================================

with st.form("credit_scoring_form"):

    st.markdown(
        '<div class="section-header">Financial Profile</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        checking_label = st.selectbox(
            "Checking Account Status",
            list(checking_account.values())
        )

        duration = st.number_input(
            "Credit Duration (months)",
            min_value=1,
            max_value=120,
            value=12
        )

        history_label = st.selectbox(
            "Credit History",
            list(credit_history.values())
        )

        purpose_label = st.selectbox(
            "Loan Purpose",
            list(purpose.values())
        )

        credit_amount = st.number_input(
            "Credit Amount",
            min_value=0,
            max_value=100000,
            value=2500,
            step=100
        )

        savings_label = st.selectbox(
            "Savings / Bonds",
            list(savings.values())
        )

        employment_label = st.selectbox(
            "Employment Duration",
            list(employment.values())
        )

    with col2:

        installment_rate = st.number_input(
            "Installment Rate (% of disposable income)",
            min_value=1,
            max_value=4,
            value=2
        )

        status_label = st.selectbox(
            "Personal Status",
            list(personal_status.values())
        )

        debtor_label = st.selectbox(
            "Other Debtors / Guarantors",
            list(debtors.values())
        )

        residence = st.number_input(
            "Current Residence (years)",
            min_value=1,
            max_value=10,
            value=2
        )

        property_label = st.selectbox(
            "Property",
            list(property_options.values())
        )

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=35
        )

    st.markdown(
        '<div class="section-header">Additional Profile</div>',
        unsafe_allow_html=True
    )

    col3, col4, col5 = st.columns(3)

    with col3:

        installment_label = st.selectbox(
            "Other Installment Plans",
            list(installment_plans.values())
        )

        existing_credits = st.number_input(
            "Existing Credits at This Bank",
            min_value=1,
            max_value=10,
            value=1
        )

    with col4:

        job_label = st.selectbox(
            "Employment / Job Type",
            list(job.values())
        )

        dependents = st.number_input(
            "People Supported",
            min_value=1,
            max_value=10,
            value=1
        )

    with col5:

        telephone_label = st.selectbox(
            "Telephone",
            list(telephone.values())
        )

        foreign_worker_label = st.selectbox(
            "Foreign Worker",
            list(foreign_worker.values())
        )

    st.divider()

    submitted = st.form_submit_button(
        "Analyze Credit Risk",
        use_container_width=True
    )


# ============================================================
# CONVERT LABELS BACK TO MODEL CODES
# ============================================================

if submitted:

    checking_code = next(
        key for key, value in checking_account.items()
        if value == checking_label
    )

    history_code = next(
        key for key, value in credit_history.items()
        if value == history_label
    )

    purpose_code = next(
        key for key, value in purpose.items()
        if value == purpose_label
    )

    savings_code = next(
        key for key, value in savings.items()
        if value == savings_label
    )

    employment_code = next(
        key for key, value in employment.items()
        if value == employment_label
    )

    status_code = next(
        key for key, value in personal_status.items()
        if value == status_label
    )

    debtor_code = next(
        key for key, value in debtors.items()
        if value == debtor_label
    )

    property_code = next(
        key for key, value in property_options.items()
        if value == property_label
    )

    installment_code = next(
        key for key, value in installment_plans.items()
        if value == installment_label
    )

    job_code = next(
        key for key, value in job.items()
        if value == job_label
    )

    telephone_code = next(
        key for key, value in telephone.items()
        if value == telephone_label
    )

    foreign_worker_code = next(
        key for key, value in foreign_worker.items()
        if value == foreign_worker_label
    )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    customer_data = {
        "Attribute1": checking_code,
        "Attribute2": duration,
        "Attribute3": history_code,
        "Attribute4": purpose_code,
        "Attribute5": credit_amount,
        "Attribute6": savings_code,
        "Attribute7": employment_code,
        "Attribute8": installment_rate,
        "Attribute9": status_code,
        "Attribute10": debtor_code,
        "Attribute11": residence,
        "Attribute12": property_code,
        "Attribute13": age,
        "Attribute14": installment_code,
        "Attribute15": next(
            key for key, value in housing.items()
            if value == housing_label
        )
        if False else "A152",
        "Attribute16": existing_credits,
        "Attribute17": job_code,
        "Attribute18": dependents,
        "Attribute19": telephone_code,
        "Attribute20": foreign_worker_code
    }


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        customer_df = pd.DataFrame(
            [customer_data]
        )

        prediction = model.predict(
            customer_df
        )[0]

        probabilities = model.predict_proba(
            customer_df
        )[0]

        bad_probability = probabilities[0]
        good_probability = probabilities[1]


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-header">Credit Assessment</div>',
            unsafe_allow_html=True
        )

        result1, result2, result3 = st.columns(3)

        with result1:

            if prediction == 1:
                st.success("GOOD CREDIT")
            else:
                st.error("BAD CREDIT")

        with result2:

            st.metric(
                "Good Credit Probability",
                f"{good_probability:.1%}"
            )

        with result3:

            st.metric(
                "Bad Credit Probability",
                f"{bad_probability:.1%}"
            )


        st.write("Model confidence")

        st.progress(
            float(max(
                good_probability,
                bad_probability
            ))
        )


        if prediction == 1:

            st.info(
                "Based on the information provided, "
                "the model classifies this applicant "
                "as a Good Credit risk."
            )

        else:

            st.warning(
                "Based on the information provided, "
                "the model classifies this applicant "
                "as a Bad Credit risk."
            )


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


st.markdown("---")

# Model Performance
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test ROC-AUC", "0.7861")

with col2:
    st.metric("CV ROC-AUC", "0.7990")

with col3:
    st.metric("Test Accuracy", "73.00%")

st.markdown(
    """
    The Random Forest model was selected after comparing multiple
    classification algorithms and performing hyperparameter tuning.
    """
)

# Models Comparison
st.subheader("Models Evaluated")

performance_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Cross-Validation ROC-AUC": [
        0.7882,
        0.6074,
        0.7930
    ]
}

performance_df = pd.DataFrame(performance_data)

st.dataframe(
    performance_df,
    use_container_width=True,
    hide_index=True
)

# How it works
st.subheader("How CreditScore AI Works")

st.markdown(
    """
    **1. Customer Information**  
    Financial and demographic information is entered through the form.

    **2. Data Preprocessing**  
    Numerical features are standardized and categorical features are
    converted using one-hot encoding.

    **3. Machine Learning Model**  
    A tuned Random Forest classifier analyzes the processed information.

    **4. Risk Prediction**  
    The model estimates whether the customer belongs to the Good Credit
    or Bad Credit category.

    **5. Probability Score**  
    The application also displays the model's estimated probability.
    """
)

# Disclaimer
st.markdown("---")

st.caption(
    "Disclaimer: This application is an educational machine-learning "
    "project based on the UCI German Credit dataset. It should not be "
    "used as a real-world financial or lending decision system."
)

st.caption("CreditScore AI | CodeAlpha Machine Learning Internship")