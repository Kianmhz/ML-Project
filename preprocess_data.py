import pandas as pd

# Read the data
df = pd.read_csv('/Users/kianmhz/Desktop/ML Project/insurance_data.csv')

# Convert date columns to datetime
df['POLICY_EFF_DT'] = pd.to_datetime(df['POLICY_EFF_DT'])
df['LOSS_DT'] = pd.to_datetime(df['LOSS_DT'])
df['REPORT_DT'] = pd.to_datetime(df['REPORT_DT'])

# Encode CLAIM_STATUS: A = 0, D = 1
df['CLAIM_STATUS'] = df['CLAIM_STATUS'].map({'A': 0, 'D': 1})

# Numerical Features (keep as-is)
numerical_features = [
    'PREMIUM_AMOUNT',
    'CLAIM_AMOUNT',
    'AGE',
    'TENURE',
    'NO_OF_FAMILY_MEMBERS',
    'INCIDENT_HOUR_OF_THE_DAY'
]

# Engineered Numerical Features
df['CLAIM_TO_PREMIUM_RATIO'] = df['CLAIM_AMOUNT'] / df['PREMIUM_AMOUNT']
df['REPORT_DELAY_DAYS'] = (df['REPORT_DT'] - df['LOSS_DT']).dt.days
df['POLICY_DURATION_DAYS'] = (df['LOSS_DT'] - df['POLICY_EFF_DT']).dt.days

# Categorical Features
categorical_features = [
    'INSURANCE_TYPE',
    'MARITAL_STATUS',
    'EMPLOYMENT_STATUS',
    'RISK_SEGMENTATION',
    'HOUSE_TYPE',
    'SOCIAL_CLASS',
    'CUSTOMER_EDUCATION_LEVEL',
    'INCIDENT_SEVERITY',
    'AUTHORITY_CONTACTED',
    'ANY_INJURY',
    'POLICE_REPORT_AVAILABLE',
    'INCIDENT_STATE'
]

# Select final columns
final_columns = (
    ['CLAIM_STATUS'] +
    numerical_features +
    ['CLAIM_TO_PREMIUM_RATIO', 'REPORT_DELAY_DAYS', 'POLICY_DURATION_DAYS'] +
    categorical_features
)

# Create final dataframe with selected columns
df_processed = df[final_columns]

# Save to CSV
df_processed.to_csv('/Users/kianmhz/Desktop/ML Project/preprocessed_data.csv', index=False)

print(f"Preprocessed data saved. Shape: {df_processed.shape}")
print(f"\nColumns: {list(df_processed.columns)}")
print(f"\nFirst few rows:")
print(df_processed.head())
