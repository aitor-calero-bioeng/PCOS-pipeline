
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. LOAD AND MERGE DATASETS

print("Loading data files...")

excel_path = "C:/Users/pilar/OneDrive/AITOR/Bioingeniería/Personal Proyects/PCOS_data_without_infertility.xlsx"
csv_path = "C:/Users/pilar/OneDrive/AITOR/Bioingeniería/Personal Proyects/PCOS_infertility.csv"

# Load Excel sheet and CSV file
df_no_inf = pd.read_excel(excel_path, sheet_name="Full_new")
df_inf = pd.read_csv(csv_path)

# Strip hidden whitespace in column names
df_no_inf.columns = [c.strip() for c in df_no_inf.columns]
df_inf.columns = [c.strip() for c in df_inf.columns]

# Fix Patient ID discrepancy by adding 10000 to the Excel IDs
df_no_inf['Patient File No.'] = df_no_inf['Patient File No.'].astype('int64') + 10000

# Left join on Patient File No.
df_full = pd.merge(df_no_inf, df_inf, on='Patient File No.', suffixes=('', '_dup'), how='left')

# Drop duplicated columns created by the merge
cols_to_drop = [
    'Unnamed: 44', 
    'Sl. No_dup', 
    'PCOS (Y/N)_dup', 
    'I beta-HCG(mIU/mL)_dup',
    'II beta-HCG(mIU/mL)_dup', 
    'AMH(ng/mL)_dup'
]
df_full = df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns], errors='ignore')

print(f"Merge successful! Total rows: {df_full.shape[0]}, Total columns: {df_full.shape[1]}")


# 2. DATA CLEANING & PREPROCESSING

print("Cleaning data and converting column types...")

# Case-insensitive column search to prevent KeyErrors
col_amh = [c for c in df_full.columns if 'amh' in c.lower()][0]
col_beta2 = [c for c in df_full.columns if 'ii' in c.lower() and 'beta' in c.lower()][0]

# Convert columns stored as strings into numeric types
df_full[col_amh] = pd.to_numeric(df_full[col_amh], errors='coerce')
df_full[col_beta2] = pd.to_numeric(df_full[col_beta2], errors='coerce')

# Impute missing values using column medians
cols_with_na = ['Marraige Status (Yrs)', col_beta2, col_amh, 'Fast food (Y/N)']
for c in cols_with_na:
    if c in df_full.columns:
        median_val = df_full[c].median()
        df_full[c] = df_full[c].fillna(median_val)

# Sanitize column names by replacing spaces and parentheses
df_full.columns = [
    c.replace(' ', '_').replace(':', '_').replace('(', '_').replace(')', '_') 
    for c in df_full.columns
]


# 3. FEATURE SELECTION & TRAIN/TEST SPLIT

print("Selecting clinical features and splitting data...")

# Target variable (PCOS diagnosis)
target_col = [c for c in df_full.columns if 'pcos' in c.lower()][0]
y = df_full[target_col]

# Select key predictor features
col_age = [c for c in df_full.columns if 'age' in c.lower()][0]
col_bmi = [c for c in df_full.columns if 'bmi' in c.lower()][0]
col_cycle = [c for c in df_full.columns if 'cycle' in c.lower()][0]
col_follicle_l = [c for c in df_full.columns if 'follicle' in c.lower() and 'l' in c.lower()][0]
col_follicle_r = [c for c in df_full.columns if 'follicle' in c.lower() and 'r' in c.lower()][0]
col_weight = [c for c in df_full.columns if 'weight' in c.lower() and 'gain' in c.lower()][0]
col_hair = [c for c in df_full.columns if 'hair' in c.lower()][0]
col_amh_clean = [c for c in df_full.columns if 'amh' in c.lower()][0]

feature_cols = [
    col_age, col_bmi, col_cycle, 
    col_follicle_l, col_follicle_r, 
    col_weight, col_hair, col_amh_clean
]

X = df_full[feature_cols]

# 80% Train / 20% Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print(f"Training samples (80%): {X_train.shape[0]}")
print(f"Testing samples (20%): {X_test.shape[0]}")


# 4. MODEL TRAINING & EVALUATION

print("Training Random Forest Classifier...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy on test set: {accuracy:.2%}\n")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Export model
os.makedirs("models", exist_ok=True)
model_path = "models/rf_pcos_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved successfully as '{model_path}'")