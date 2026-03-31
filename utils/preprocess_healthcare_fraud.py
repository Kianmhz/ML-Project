import os
import pandas as pd
import numpy as np

base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir   = os.path.join(base_dir, 'dataset', 'healthcare_fraud')
output_path = os.path.join(base_dir, 'dataset', 'healthcare_fraud_preprocessed.csv')

# ============================================================
# LOAD RAW FILES
# ============================================================

print("Loading raw files...")
labels   = pd.read_csv(os.path.join(data_dir, 'Train-1542865627584.csv'))
bene     = pd.read_csv(os.path.join(data_dir, 'Train_Beneficiarydata-1542865627584.csv'))
inpat    = pd.read_csv(os.path.join(data_dir, 'Train_Inpatientdata-1542865627584.csv'))
outpat   = pd.read_csv(os.path.join(data_dir, 'Train_Outpatientdata-1542865627584.csv'))

print(f"  Labels:      {labels.shape[0]:>7,} rows  ({labels['PotentialFraud'].eq('Yes').sum()} fraud providers)")
print(f"  Beneficiary: {bene.shape[0]:>7,} rows")
print(f"  Inpatient:   {inpat.shape[0]:>7,} rows")
print(f"  Outpatient:  {outpat.shape[0]:>7,} rows\n")

# ============================================================
# BENEFICIARY FEATURE ENGINEERING
# ============================================================

print("Engineering beneficiary features...")

# Parse dates
bene['DOB'] = pd.to_datetime(bene['DOB'])
bene['DOD'] = pd.to_datetime(bene['DOD'], errors='coerce')

# Age as of end of 2009 (all claims are 2009)
reference_date        = pd.Timestamp('2009-12-31')
bene['Age']           = ((reference_date - bene['DOB']).dt.days / 365.25).astype(int)
bene['Is_Deceased']   = bene['DOD'].notna().astype(int)

# Chronic conditions: 1 = has condition, 2 = does not — convert to 0/1
chronic_cols = [c for c in bene.columns if c.startswith('ChronicCond_')]
for col in chronic_cols:
    bene[col] = (bene[col] == 1).astype(int)
bene['Chronic_Count'] = bene[chronic_cols].sum(axis=1)

# Renal disease: "Y" = yes, "0" = no
bene['RenalDiseaseIndicator'] = (bene['RenalDiseaseIndicator'] == 'Y').astype(int)

print(f"  Beneficiary features ready. Age range: {bene['Age'].min()}–{bene['Age'].max()}\n")

# ============================================================
# CLAIM-LEVEL FEATURE ENGINEERING
# ============================================================

print("Engineering claim-level features...")

# --- Inpatient ---
date_cols_ip = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']
for col in date_cols_ip:
    inpat[col] = pd.to_datetime(inpat[col])

inpat['ClaimDuration']  = (inpat['ClaimEndDt']   - inpat['ClaimStartDt']).dt.days
inpat['HospitalStay']   = (inpat['DischargeDt']  - inpat['AdmissionDt']).dt.days
inpat['SamePhysician']  = (
    inpat['AttendingPhysician'].notna() &
    (inpat['AttendingPhysician'] == inpat['OperatingPhysician'])
).astype(int)

diag_cols_ip = [c for c in inpat.columns if c.startswith('ClmDiagnosisCode_')]
proc_cols_ip = [c for c in inpat.columns if c.startswith('ClmProcedureCode_')]
inpat['DiagnosisCount']  = inpat[diag_cols_ip].notna().sum(axis=1)
inpat['ProcedureCount']  = inpat[proc_cols_ip].notna().sum(axis=1)
inpat['ClaimType']       = 'IP'

# --- Outpatient ---
date_cols_op = ['ClaimStartDt', 'ClaimEndDt']
for col in date_cols_op:
    outpat[col] = pd.to_datetime(outpat[col])

outpat['ClaimDuration'] = (outpat['ClaimEndDt'] - outpat['ClaimStartDt']).dt.days
outpat['HospitalStay']  = np.nan   # outpatient has no admission/discharge
outpat['SamePhysician'] = (
    outpat['AttendingPhysician'].notna() &
    (outpat['AttendingPhysician'] == outpat['OperatingPhysician'])
).astype(int)

diag_cols_op = [c for c in outpat.columns if c.startswith('ClmDiagnosisCode_')]
proc_cols_op = [c for c in outpat.columns if c.startswith('ClmProcedureCode_')]
outpat['DiagnosisCount'] = outpat[diag_cols_op].notna().sum(axis=1)
outpat['ProcedureCount'] = outpat[proc_cols_op].notna().sum(axis=1)
outpat['ClaimType']      = 'OP'

# Shared columns for combining
shared_cols = [
    'BeneID', 'ClaimID', 'Provider', 'ClaimType',
    'ClaimStartDt', 'ClaimEndDt', 'ClaimDuration', 'HospitalStay',
    'InscClaimAmtReimbursed', 'DeductibleAmtPaid',
    'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician',
    'DiagnosisCount', 'ProcedureCount', 'SamePhysician',
]
claims = pd.concat([inpat[shared_cols], outpat[shared_cols]], ignore_index=True)
print(f"  Combined claims: {len(claims):,} rows\n")

# ============================================================
# JOIN BENEFICIARY DATA ONTO CLAIMS
# ============================================================

print("Joining beneficiary data to claims...")

bene_features = [
    'BeneID', 'Age', 'Is_Deceased', 'Chronic_Count', 'RenalDiseaseIndicator',
    'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
    'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
] + chronic_cols

claims = claims.merge(bene[bene_features], on='BeneID', how='left')
print(f"  Claims after join: {len(claims):,} rows\n")

# ============================================================
# AGGREGATE TO PROVIDER LEVEL
# ============================================================

print("Aggregating to provider level...")

ip_claims = claims[claims['ClaimType'] == 'IP']
op_claims = claims[claims['ClaimType'] == 'OP']

# --- Volume & money ---
prov = claims.groupby('Provider').agg(
    total_claims            = ('ClaimID',               'count'),
    ip_claims               = ('ClaimType',             lambda x: (x == 'IP').sum()),
    op_claims               = ('ClaimType',             lambda x: (x == 'OP').sum()),
    total_reimbursed        = ('InscClaimAmtReimbursed','sum'),
    avg_claim_reimbursed    = ('InscClaimAmtReimbursed','mean'),
    max_claim_reimbursed    = ('InscClaimAmtReimbursed','max'),
    total_deductible        = ('DeductibleAmtPaid',     'sum'),
    avg_claim_duration      = ('ClaimDuration',         'mean'),
    max_claim_duration      = ('ClaimDuration',         'max'),
    unique_benes            = ('BeneID',                'nunique'),
    unique_attending        = ('AttendingPhysician',    'nunique'),
    unique_operating        = ('OperatingPhysician',    'nunique'),
    pct_same_physician      = ('SamePhysician',         'mean'),
    avg_diagnosis_count     = ('DiagnosisCount',        'mean'),
    avg_procedure_count     = ('ProcedureCount',        'mean'),
).reset_index()

# Inpatient-only aggregations
ip_agg = ip_claims.groupby('Provider').agg(
    avg_hospital_stay   = ('HospitalStay', 'mean'),
    max_hospital_stay   = ('HospitalStay', 'max'),
).reset_index()

prov = prov.merge(ip_agg, on='Provider', how='left')

# --- Derived ratios ---
prov['ip_ratio']            = prov['ip_claims'] / prov['total_claims']
prov['claims_per_bene']     = prov['total_claims'] / prov['unique_benes']
prov['reimbursed_per_bene'] = prov['total_reimbursed'] / prov['unique_benes']
prov['physicians_per_bene'] = prov['unique_attending'] / prov['unique_benes']

# --- Beneficiary demographics per provider ---
bene_agg = claims.groupby('Provider').agg(
    avg_bene_age            = ('Age',                     'mean'),
    pct_deceased            = ('Is_Deceased',             'mean'),
    avg_chronic_count       = ('Chronic_Count',           'mean'),
    pct_renal_disease       = ('RenalDiseaseIndicator',   'mean'),
    avg_ip_annual_reimb     = ('IPAnnualReimbursementAmt','mean'),
    avg_op_annual_reimb     = ('OPAnnualReimbursementAmt','mean'),
    avg_months_partA        = ('NoOfMonths_PartACov',     'mean'),
    avg_months_partB        = ('NoOfMonths_PartBCov',     'mean'),
).reset_index()

# Per-chronic-condition prevalence among a provider's patients
for col in chronic_cols:
    bene_agg[f'pct_{col}'] = claims.groupby('Provider')[col].mean().values

prov = prov.merge(bene_agg, on='Provider', how='left')

# Fill NaN from inpatient-only cols for providers with no IP claims
prov['avg_hospital_stay'] = prov['avg_hospital_stay'].fillna(0)
prov['max_hospital_stay'] = prov['max_hospital_stay'].fillna(0)

print(f"  Provider-level features: {prov.shape[1] - 1} features across {prov.shape[0]:,} providers\n")

# ============================================================
# MERGE LABELS
# ============================================================

labels['PotentialFraud'] = (labels['PotentialFraud'] == 'Yes').astype(int)
prov = prov.merge(labels, on='Provider', how='inner')

fraud_count = prov['PotentialFraud'].sum()
print(f"Final dataset: {len(prov):,} providers  |  {fraud_count} fraud ({fraud_count/len(prov)*100:.1f}%)\n")

# ============================================================
# SAVE
# ============================================================

prov.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Shape: {prov.shape}")
print(f"\nColumns:\n{list(prov.columns)}")