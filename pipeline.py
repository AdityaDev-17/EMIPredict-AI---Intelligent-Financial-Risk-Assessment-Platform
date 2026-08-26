import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb


class EMIPredictionPipeline:
    def __init__(self, artifacts_dir='artifacts'):
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(f'{artifacts_dir}/classifier.json')

        self.regressor = xgb.XGBRegressor()
        self.regressor.load_model(f'{artifacts_dir}/regressor.json')
        self.ohe = joblib.load(f'{artifacts_dir}/onehot_encoder.pkl')
        self.target_encoder = joblib.load(f'{artifacts_dir}/target_encoder.pkl')

        with open(f'{artifacts_dir}/metadata.json') as f:
            self.metadata = json.load(f)

        with open(f'{artifacts_dir}/impute_values.json') as f:
            self.impute_values = json.load(f)

        self.tree_features = self.metadata['tree_features']
        self.nominal_cols = self.metadata['nominal_cols']
        self.education_order = self.metadata['education_order']

    def _impute(self, df):
        for col, val in self.impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _engineer_features(self, df):
        df['total_monthly_expenses'] = (
            df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
            df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses'] +
            df['current_emi_amount']
        )
        df['loan_to_income_ratio'] = df['requested_amount'] / df['monthly_salary']
        df['emi_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
        df['expense_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary']
        df['disposable_income'] = df['monthly_salary'] - df['total_monthly_expenses']

        df['credit_risk_component'] = (850 - df['credit_score']) / (850 - 300)
        df['existing_loan_flag'] = (df['existing_loans'] == 'Yes').astype(int)
        df['employment_stability'] = df['years_of_employment'].clip(upper=10) / 10
        df['risk_score'] = (
            0.4 * df['credit_risk_component'] +
            0.3 * df['existing_loan_flag'] +
            0.3 * (1 - df['employment_stability'])
        )

        df['log_loan_to_income_ratio'] = np.log1p(df['loan_to_income_ratio'])
        df['log_expense_to_income_ratio'] = np.log1p(df['expense_to_income_ratio'])

        df['education_encoded'] = df['education'].map(self.education_order)
        return df

    def _encode_categoricals(self, df):
        encoded = self.ohe.transform(df[self.nominal_cols])
        encoded_df = pd.DataFrame(
            encoded, columns=self.ohe.get_feature_names_out(self.nominal_cols), index=df.index
        )
        return pd.concat([df, encoded_df], axis=1)

    def preprocess(self, raw_input: dict) -> pd.DataFrame:
        df = pd.DataFrame([raw_input])
        df = self._impute(df)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df)

        # Ensure every expected tree_feature column exists (missing one-hot cols default to 0)
        for col in self.tree_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.tree_features]

    def predict(self, raw_input: dict) -> dict:
        X = self.preprocess(raw_input)

        eligibility_encoded = self.classifier.predict(X)[0]
        eligibility_proba = self.classifier.predict_proba(X)[0]
        eligibility_label = self.target_encoder.inverse_transform([eligibility_encoded])[0]

        max_emi = self.regressor.predict(X)[0]

        return {
            'eligibility': eligibility_label,
            'eligibility_probabilities': dict(zip(self.target_encoder.classes_, eligibility_proba.round(3))),
            'max_monthly_emi': round(float(max_emi), 2)
        }