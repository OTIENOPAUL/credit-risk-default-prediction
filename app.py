import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Home Credit Underwriting Portal",
    layout="wide"
)

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('credit_risk_lgb_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, model_features = load_artifacts()

# --- HEADER SECTION ---
st.title("Automated Credit Risk Underwriting Portal")
st.markdown("""
This interactive portal simulates a production-grade credit scoring engine. 
Adjust the applicant's financial indicators in the sidebar to evaluate their **Probability of Default (PD)** in real time.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Applicant Profile Inputs")

def user_input_parameters():
    ext_source_2 = st.sidebar.slider("External Credit Score 2 (EXT_SOURCE_2)", 0.0, 1.0, 0.5, 0.01)
    ext_source_3 = st.sidebar.slider("External Credit Score 3 (EXT_SOURCE_3)", 0.0, 1.0, 0.5, 0.01)
    ext_source_1 = st.sidebar.slider("External Credit Score 1 (EXT_SOURCE_1)", 0.0, 1.0, 0.5, 0.01)
    
    amt_income = st.sidebar.number_input("Annual Income (AMT_INCOME_TOTAL)", min_value=10000, max_value=5000000, value=150000, step=5000)
    amt_credit = st.sidebar.number_input("Requested Loan Amount (AMT_CREDIT)", min_value=10000, max_value=4000000, value=500000, step=10000)
    amt_annuity = st.sidebar.number_input("Monthly Debt Annuity (AMT_ANNUITY)", min_value=500, max_value=200000, value=25000, step=500)
    
    years_employed = st.sidebar.slider("Employment Duration (Years)", 0, 40, 5)
    days_employed = -1 * (years_employed * 365)
    
    age_years = st.sidebar.slider("Applicant Age (Years)", 18, 70, 35)
    days_birth = -1 * (age_years * 365)

    data = {
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'EXT_SOURCE_1': ext_source_1,
        'AMT_INCOME_TOTAL': amt_income,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'DAYS_EMPLOYED': days_employed,
        'DAYS_BIRTH': days_birth
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_parameters()

# --- ALIGN INPUTS WITH MODEL SCHEMA ---
# 1. Create a dataframe matching all features the model was trained on, initialized to 0
full_input_df = pd.DataFrame(columns=model_features)
full_input_df.loc[0] = 0 

# 2. Populate with user inputs where available
for col in input_df.columns:
    if col in full_input_df.columns:
        full_input_df[col] = input_df[col]

# 3. Ensure strict column ordering matching model_features exactly
full_input_df = full_input_df[model_features]

# --- PREDICTION LOGIC ---
if st.sidebar.button("Run Credit Assessment 🚀", type="primary"):
    # Generate prediction probability by passing a NumPy array to bypass pandas categorical checks
    default_probability = model.predict(full_input_df.to_numpy())[0]
    threshold = 0.12
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scoring Output")
        st.metric(label="Predicted Probability of Default (PD)", value=f"{default_probability * 100:.2f}%")
        
        if default_probability < threshold:
            st.success("**Decision: APPROVED** — Risk profile is within acceptable institutional tolerance.")
        else:
            st.error("**Decision: REJECTED / MANUAL REVIEW** — Predicted default risk exceeds allowable threshold.")

    with col2:
        st.subheader("Underwriting Insights")
        st.markdown(f"""
        * **Risk Threshold Limit:** {threshold * 100}%
        * **Loan-to-Income Ratio:** {input_df['AMT_CREDIT'].values[0] / input_df['AMT_INCOME_TOTAL'].values[0]:.2f}x
        * **External Score Health:** Ext 2: `{input_df['EXT_SOURCE_2'].values[0]}` | Ext 3: `{input_df['EXT_SOURCE_3'].values[0]}`
        """)
else:
    st.warning("👈 Adjust applicant parameters in the sidebar and click **Run Credit Assessment** to view the prediction.")