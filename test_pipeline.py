from pipeline import EMIPredictionPipeline

pipeline = EMIPredictionPipeline(artifacts_dir='artifacts')

sample_input = {
    'age': 35, 'gender': 'Male', 'marital_status': 'Married', 'education': 'Graduate',
    'monthly_salary': 55000, 'employment_type': 'Private', 'years_of_employment': 4.5,
    'company_type': 'Mid-size', 'house_type': 'Own', 'monthly_rent': 0,
    'family_size': 3, 'dependents': 2, 'school_fees': 3000, 'college_fees': 0,
    'travel_expenses': 4000, 'groceries_utilities': 12000, 'other_monthly_expenses': 6000,
    'existing_loans': 'No', 'current_emi_amount': 0, 'credit_score': 720,
    'bank_balance': 200000, 'emergency_fund': 80000, 'emi_scenario': 'Vehicle EMI',
    'requested_amount': 300000, 'requested_tenure': 36
}

result = pipeline.predict(sample_input)
print(result)