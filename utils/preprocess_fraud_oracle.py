import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv('/Users/kianmhz/Desktop/ML-Project/fraud_oracle.csv')

print("=" * 60)
print("FRAUD ORACLE DATA - PREPROCESSING")
print("=" * 60)
print(f"\nOriginal Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# ============================================================
# CHECK FOR MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

# ============================================================
# TARGET VARIABLE
# ============================================================

# FraudFound_P is already binary (0/1)
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df['FraudFound_P'].value_counts())
print(f"\nFraud rate: {df['FraudFound_P'].mean()*100:.2f}%")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# 1. Convert categorical ordinal features to numerical

# Age mapping
age_mapping = {
    '0': 0, '16 to 17': 16.5, '18 to 20': 19, '21 to 25': 23, '26 to 30': 28,
    '31 to 35': 33, '36 to 40': 38, '41 to 50': 45.5, '51 to 65': 58, 'over 65': 70
}
df['Age_Numeric'] = df['Age'].map(age_mapping).fillna(df['Age'].apply(lambda x: 0 if x == '0' else 35))
print("Added: Age_Numeric")

# AgeOfPolicyHolder mapping
policy_holder_age_mapping = {
    '16 to 17': 16.5, '18 to 20': 19, '21 to 25': 23, '26 to 30': 28,
    '31 to 35': 33, '36 to 40': 38, '41 to 50': 45.5, '51 to 65': 58, 'over 65': 70
}
df['PolicyHolderAge_Numeric'] = df['AgeOfPolicyHolder'].map(policy_holder_age_mapping).fillna(35)
print("Added: PolicyHolderAge_Numeric")

# Vehicle Price mapping
price_mapping = {
    'less than 20000': 15000,
    '20000 to 29000': 24500,
    '30000 to 39000': 34500,
    '40000 to 59000': 49500,
    '60000 to 69000': 64500,
    'more than 69000': 80000
}
df['VehiclePrice_Numeric'] = df['VehiclePrice'].map(price_mapping).fillna(30000)
print("Added: VehiclePrice_Numeric")

# Age of Vehicle mapping
vehicle_age_mapping = {
    'new': 0, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
    '6 years': 6, '7 years': 7, 'more than 7': 10
}
df['VehicleAge_Numeric'] = df['AgeOfVehicle'].map(vehicle_age_mapping).fillna(4)
print("Added: VehicleAge_Numeric")

# Days Policy Accident mapping
days_policy_acc_mapping = {
    'none': 0, '1 to 7': 4, '8 to 15': 11.5, '15 to 30': 22.5, 'more than 30': 45
}
df['DaysPolicyAccident_Numeric'] = df['Days_Policy_Accident'].map(days_policy_acc_mapping).fillna(45)
print("Added: DaysPolicyAccident_Numeric")

# Days Policy Claim mapping
days_policy_claim_mapping = {
    'none': 0, '1 to 7': 4, '8 to 15': 11.5, '15 to 30': 22.5, 'more than 30': 45
}
df['DaysPolicyClaim_Numeric'] = df['Days_Policy_Claim'].map(days_policy_claim_mapping).fillna(45)
print("Added: DaysPolicyClaim_Numeric")

# Past Number of Claims mapping
past_claims_mapping = {
    'none': 0, '1': 1, '2 to 4': 3, 'more than 4': 6
}
df['PastClaims_Numeric'] = df['PastNumberOfClaims'].map(past_claims_mapping).fillna(0)
print("Added: PastClaims_Numeric")

# Number of Supplements mapping
supplements_mapping = {
    'none': 0, '1 to 2': 1.5, '3 to 5': 4, 'more than 5': 7
}
df['NumSupplements_Numeric'] = df['NumberOfSuppliments'].map(supplements_mapping).fillna(0)
print("Added: NumSupplements_Numeric")

# Address Change Claim mapping
address_change_mapping = {
    'no change': 0, 'under 6 months': 0.25, '1 year': 1, '2 to 3 years': 2.5, '4 to 8 years': 6
}
df['AddressChangeClaim_Numeric'] = df['AddressChange_Claim'].map(address_change_mapping).fillna(0)
print("Added: AddressChangeClaim_Numeric")

# Number of Cars mapping
num_cars_mapping = {
    '1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3.5, '5 to 8': 6.5, 'more than 8': 10
}
df['NumCars_Numeric'] = df['NumberOfCars'].map(num_cars_mapping).fillna(1)
print("Added: NumCars_Numeric")

# 2. Binary encoding for Yes/No columns
df['PoliceReportFiled_Binary'] = df['PoliceReportFiled'].map({'Yes': 1, 'No': 0}).fillna(0)
df['WitnessPresent_Binary'] = df['WitnessPresent'].map({'Yes': 1, 'No': 0}).fillna(0)
print("Added: PoliceReportFiled_Binary, WitnessPresent_Binary")

# 3. Create derived features

# Age difference between driver and vehicle
df['Driver_Vehicle_Age_Diff'] = df['Age_Numeric'] - df['VehicleAge_Numeric']
print("Added: Driver_Vehicle_Age_Diff")

# Policy holder vs driver age difference
df['PolicyHolder_Driver_Age_Diff'] = df['PolicyHolderAge_Numeric'] - df['Age_Numeric']
print("Added: PolicyHolder_Driver_Age_Diff")

# Days between accident and claim
df['Days_Accident_to_Claim'] = df['DaysPolicyClaim_Numeric'] - df['DaysPolicyAccident_Numeric']
print("Added: Days_Accident_to_Claim")

# Risk indicators
df['Has_Past_Claims'] = (df['PastClaims_Numeric'] > 0).astype(int)
df['New_Vehicle'] = (df['AgeOfVehicle'] == 'new').astype(int)
df['Young_Driver'] = (df['Age_Numeric'] < 25).astype(int)
df['High_Value_Vehicle'] = (df['VehiclePrice_Numeric'] > 50000).astype(int)
df['External_Agent'] = (df['AgentType'] == 'External').astype(int)
df['Is_Urban'] = (df['AccidentArea'] == 'Urban').astype(int)
df['Policy_Holder_Fault'] = (df['Fault'] == 'Policy Holder').astype(int)
print("Added: Has_Past_Claims, New_Vehicle, Young_Driver, High_Value_Vehicle, External_Agent, Is_Urban, Policy_Holder_Fault")

# Encode month as numeric
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df['Month_Numeric'] = df['Month'].map(month_mapping).fillna(6)
df['MonthClaimed_Numeric'] = df['MonthClaimed'].map(month_mapping).fillna(6)
print("Added: Month_Numeric, MonthClaimed_Numeric")

# Encode day of week
day_mapping = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
    'Friday': 5, 'Saturday': 6, 'Sunday': 7
}
df['DayOfWeek_Numeric'] = df['DayOfWeek'].map(day_mapping).fillna(3)
df['DayOfWeekClaimed_Numeric'] = df['DayOfWeekClaimed'].map(day_mapping).fillna(3)
print("Added: DayOfWeek_Numeric, DayOfWeekClaimed_Numeric")

# Weekend indicators
df['Accident_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
df['Claim_Weekend'] = df['DayOfWeekClaimed'].isin(['Saturday', 'Sunday']).astype(int)
print("Added: Accident_Weekend, Claim_Weekend")

# ============================================================
# DEFINE FEATURE TYPES
# ============================================================

numerical_features = [
    'Age_Numeric', 'PolicyHolderAge_Numeric', 'VehiclePrice_Numeric', 'VehicleAge_Numeric',
    'DaysPolicyAccident_Numeric', 'DaysPolicyClaim_Numeric', 'PastClaims_Numeric',
    'NumSupplements_Numeric', 'AddressChangeClaim_Numeric', 'NumCars_Numeric',
    'Deductible', 'DriverRating', 'WeekOfMonth', 'WeekOfMonthClaimed', 'Year',
    'Driver_Vehicle_Age_Diff', 'PolicyHolder_Driver_Age_Diff', 'Days_Accident_to_Claim',
    'Month_Numeric', 'MonthClaimed_Numeric', 'DayOfWeek_Numeric', 'DayOfWeekClaimed_Numeric',
    'PoliceReportFiled_Binary', 'WitnessPresent_Binary',
    'Has_Past_Claims', 'New_Vehicle', 'Young_Driver', 'High_Value_Vehicle',
    'External_Agent', 'Is_Urban', 'Policy_Holder_Fault', 'Accident_Weekend', 'Claim_Weekend'
]

categorical_features = [
    'Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault',
    'PolicyType', 'VehicleCategory', 'AgentType', 'BasePolicy'
]

# ============================================================
# CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Convert numerical columns that might be stored as strings
for col in numerical_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Convert categorical columns to string type
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype(str)

print("Numerical features cleaned and NaN values filled with median")
print("Categorical features cleaned and NaN values filled with 'Unknown'")

# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total features: {len(numerical_features) + len(categorical_features)}")

# ============================================================
# SAVE PREPROCESSED DATA
# ============================================================

# Select final columns
final_columns = ['FraudFound_P'] + numerical_features + categorical_features
df_processed = df[[col for col in final_columns if col in df.columns]]

# Save
df_processed.to_csv('/Users/kianmhz/Desktop/ML-Project/fraud_oracle_preprocessed.csv', index=False)

print(f"\nPreprocessed data saved: fraud_oracle_preprocessed.csv")
print(f"Final shape: {df_processed.shape}")
print(f"\nFinal columns: {list(df_processed.columns)}")

# Show sample
print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)
print(df_processed.head())

# ============================================================
# SAVE CLASS DISTRIBUTION
# ============================================================
import json

distribution = df_processed['FraudFound_P'].value_counts().to_dict()
with open('/Users/kianmhz/Desktop/ML-Project/distribution_oracle.json', 'w') as f:
    json.dump(distribution, f)
print(f"\nClass distribution saved to distribution_oracle.json")
print(f"Distribution: {distribution}")
