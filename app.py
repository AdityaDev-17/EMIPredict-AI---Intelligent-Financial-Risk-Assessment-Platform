import json

import pandas as pd
import streamlit as st

from pipeline import EMIPredictionPipeline

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return EMIPredictionPipeline(artifacts_dir='artifacts')


@st.cache_data
def load_eda_sample():
    return pd.read_csv('data/eda_sample.csv')


@st.cache_data
def load_model_metrics():
    with open('model_metrics.json') as f:
        return json.load(f)


# ------------------------------------------------------------------
# Page 1: Real-time Prediction
# ------------------------------------------------------------------
def page_predict(pipeline):
    st.title("💳 EMIPredict AI")
    st.caption("Intelligent Financial Risk Assessment Platform")
    st.header("Real-Time EMI Eligibility & Amount Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=75, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)

    with col2:
        st.subheader("Employment & Housing")
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=55000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=4.5, step=0.1)
        company_type = st.selectbox("Company Type", ["Large Indian", "Mid-size", "MNC", "Startup", "Small"])
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
        monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0, value=0, step=500)

    with col3:
        st.subheader("Financial Details")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0, value=200000, step=1000)
        emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, value=80000, step=1000)
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0, value=0, step=500)

    st.subheader("Monthly Expenses")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        school_fees = st.number_input("School Fees", min_value=0, value=3000, step=500)
        college_fees = st.number_input("College Fees", min_value=0, value=0, step=500)
    with exp_col2:
        travel_expenses = st.number_input("Travel Expenses", min_value=0, value=4000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities", min_value=0, value=12000, step=500)
    with exp_col3:
        other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0, value=6000, step=500)

    st.subheader("Loan Application")
    loan_col1, loan_col2, loan_col3 = st.columns(3)
    with loan_col1:
        emi_scenario = st.selectbox("EMI Scenario", [
            "E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
            "Personal Loan EMI", "Education EMI"
        ])
    with loan_col2:
        requested_amount = st.number_input("Requested Amount (INR)", min_value=1000, value=300000, step=5000)
    with loan_col3:
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, max_value=120, value=36)

    if st.button("Predict", type="primary", use_container_width=True):
        raw_input = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
            'monthly_salary': monthly_salary, 'employment_type': employment_type,
            'years_of_employment': years_of_employment, 'company_type': company_type,
            'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
            'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
            'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
            'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }

        result = pipeline.predict(raw_input)

        st.divider()
        st.subheader("Prediction Result")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            eligibility = result['eligibility']
            color = {'Eligible': 'green', 'High_Risk': 'orange', 'Not_Eligible': 'red'}[eligibility]
            st.markdown(f"### Eligibility: :{color}[{eligibility}]")
            st.write("Confidence breakdown:")
            st.bar_chart(result['eligibility_probabilities'])

        with res_col2:
            st.metric("Maximum Safe Monthly EMI", f"₹{result['max_monthly_emi']:,.2f}")


# ------------------------------------------------------------------
# Page 2: Data Exploration
# ------------------------------------------------------------------
def page_data_exploration():
    st.title("📊 Data Exploration")
    st.caption("Key patterns from the training data (3,000-row representative sample)")

    df = load_eda_sample()

    st.subheader("Eligibility Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df['emi_eligibility'].value_counts())
    with col2:
        ct = pd.crosstab(df['emi_scenario'], df['emi_eligibility'], normalize='index') * 100
        st.write("Eligibility % by EMI Scenario")
        st.dataframe(ct.round(1), use_container_width=True)

    st.divider()

    st.subheader("Loan-to-Income Ratio by Eligibility")
    st.caption("Strongest classification signal found during EDA — clear separation between classes")
    chart_data = df[['emi_eligibility', 'loan_to_income_ratio']].copy()
    chart_data['loan_to_income_ratio'] = chart_data['loan_to_income_ratio'].clip(upper=20)
    st.scatter_chart(chart_data, x='emi_eligibility', y='loan_to_income_ratio')

    st.divider()

    st.subheader("Existing Loans vs Eligibility")
    ct2 = pd.crosstab(df['existing_loans'], df['emi_eligibility'], normalize='index') * 100
    st.dataframe(ct2.round(1), use_container_width=True)
    st.caption("Applicants with existing loans show a much higher Not_Eligible rate")

    st.divider()

    st.subheader("Financial Profile by Eligibility Class")
    numeric_cols = ['monthly_salary', 'credit_score', 'bank_balance', 'emergency_fund',
                     'current_emi_amount', 'disposable_income']
    summary = df.groupby('emi_eligibility')[numeric_cols].mean().round(1)
    st.dataframe(summary, use_container_width=True)


# ------------------------------------------------------------------
# Page 3: Model Performance
# ------------------------------------------------------------------
def page_model_performance():
    st.title("📈 Model Performance Dashboard")
    st.caption("Model comparison from MLflow-tracked training runs")

    metrics = load_model_metrics()

    tab1, tab2 = st.tabs(["Classification", "Regression"])

    with tab1:
        clf = metrics['classification']
        st.success(f"Selected model: **{clf['selected_model']}** — {clf['target_met']}")

        clf_df = pd.DataFrame(clf['metrics']).set_index('model')
        st.dataframe(clf_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Accuracy by Model")
            st.bar_chart(clf_df['accuracy'])
        with col2:
            st.write("Macro F1 by Model")
            st.bar_chart(clf_df['macro_f1'])

        st.write("Per-Class F1 (XGBoost — selected model)")
        st.bar_chart(pd.Series(clf['per_class_f1_xgboost']))
        st.caption("High_Risk lags behind the other classes — it's the hardest to separate, as confirmed in EDA.")

        st.write("Top 10 Feature Importances (XGBoost)")
        fi_df = pd.DataFrame(clf['top_features']).set_index('feature')
        st.bar_chart(fi_df)

    with tab2:
        reg = metrics['regression']
        st.success(f"Selected model: **{reg['selected_model']}** — {reg['target_met']}")

        reg_df = pd.DataFrame(reg['metrics']).set_index('model')
        st.dataframe(reg_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("RMSE by Model (lower is better)")
            st.bar_chart(reg_df['rmse'])
        with col2:
            st.write("R² by Model")
            st.bar_chart(reg_df['r2'])

        st.write("Top Feature Importances (XGBoost)")
        fi_df = pd.DataFrame(reg['top_features']).set_index('feature')
        st.bar_chart(fi_df)
        st.caption("disposable_income dominates — the target is largely a function of leftover income after obligations.")


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("EMIPredict AI")
page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 Data Exploration", "📈 Model Performance"])

pipeline = load_pipeline()

if page == "🔮 Predict":
    page_predict(pipeline)
elif page == "📊 Data Exploration":
    page_data_exploration()
elif page == "📈 Model Performance":
    page_model_performance()