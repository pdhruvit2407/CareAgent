"""
===============================================================================
CAREAGENT: RANDOM FOREST MACHINE LEARNING PIPELINE (model.py)
===============================================================================
PURPOSE:
This module trains, evaluates, and serializes the primary Random Forest 
Classifier model used by CareAgent to predict 30-day hospital readmission risk.

KEY OPERATIONS:
1. Data Ingestion & Merging : Combines patient demographics, encounters, and SDOH data.
2. Categorical Encoding     : Uses LabelEncoders for categorical features (Diagnosis, Insurance).
3. Leakage-Free Split       : Performs Patient-Level Train/Test Split (80/20) to prevent data leakage.
4. Model Training           : Trains a 100-tree Random Forest Classifier.
5. Evaluation & Export      : Calculates Accuracy & ROC-AUC, outputs Feature Importance, 
                              and serializes model artifacts to src/model_artifacts.pkl.
===============================================================================
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def train_baseline_model(data_dir="data", model_dir="src"):
    """
    Loads raw healthcare datasets, performs feature engineering & encoding,
    trains the Random Forest Classifier, evaluates performance metrics, and saves model artifacts.
    """
    print("Loading data for model training...")
    
    # Load files
    patients_df = pd.read_csv(os.path.join(data_dir, "careagent_patients_5000.csv"))
    encounters_df = pd.read_csv(os.path.join(data_dir, "careagent_encounters_5000.csv"))
    sdoh_df = pd.read_csv(os.path.join(data_dir, "careagent_sdoh_5000.csv"))
    
    # Join datasets
    df = encounters_df.merge(patients_df, on="patient_id", how="left")
    df = df.merge(sdoh_df, on="patient_id", how="left")
    
    # Features & Target definition
    categorical_cols = ["sex", "insurance", "language", "encounter_type", "diagnosis_group", "sdoh_risk_level"]
    numerical_cols = ["age", "length_of_stay", "prior_encounters", "prior_ed", "prior_inpatient", "sdoh_score"]
    sdoh_flags = ["food_insecurity", "income_barrier", "housing_instability", "education_literacy_barrier", "low_social_support", "transportation_barrier"]
    
    feature_cols = categorical_cols + numerical_cols + sdoh_flags
    target_col = "readmit_30"
    
    # Encoding categorical columns
    encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # Patient-level Train-Test Split (to prevent data leakage)
    unique_patients = df_encoded["patient_id"].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    
    train_df = df_encoded[df_encoded["patient_id"].isin(train_patients)]
    test_df = df_encoded[df_encoded["patient_id"].isin(test_patients)]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    y_train_30 = train_df["readmit_30"]
    y_train_60 = train_df["readmit_60"]
    y_train_90 = train_df["readmit_90"]
    
    y_test_30 = test_df["readmit_30"]
    y_test_60 = test_df["readmit_60"]
    y_test_90 = test_df["readmit_90"]
    
    print(f"Training set size: {len(X_train)} encounters")
    print(f"Testing set size: {len(X_test)} encounters")
    
    # Train Random Forest Classifiers for each horizon
    print("Training 30-Day Risk Model...")
    rf_30 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_30.fit(X_train, y_train_30)
    
    print("Training 60-Day Risk Model...")
    rf_60 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_60.fit(X_train, y_train_60)
    
    print("Training 90-Day Risk Model...")
    rf_90 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_90.fit(X_train, y_train_90)
    
    # Predictions and Evaluation for 30-day
    y_pred_30 = rf_30.predict(X_test)
    y_prob_30 = rf_30.predict_proba(X_test)[:, 1]
    accuracy_30 = accuracy_score(y_test_30, y_pred_30)
    roc_auc_30 = roc_auc_score(y_test_30, y_prob_30)
    
    # 60-day
    y_prob_60 = rf_60.predict_proba(X_test)[:, 1]
    roc_auc_60 = roc_auc_score(y_test_60, y_prob_60)
    
    # 90-day
    y_prob_90 = rf_90.predict_proba(X_test)[:, 1]
    roc_auc_90 = roc_auc_score(y_test_90, y_prob_90)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"30-Day Model - Accuracy: {accuracy_30:.4f}, ROC-AUC: {roc_auc_30:.4f}")
    print(f"60-Day Model - ROC-AUC: {roc_auc_60:.4f}")
    print(f"90-Day Model - ROC-AUC: {roc_auc_90:.4f}")
    
    # Save artifacts
    artifacts = {
        "model": rf_30, # Backward compatibility
        "model_30": rf_30,
        "model_60": rf_60,
        "model_90": rf_90,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "sdoh_flags": sdoh_flags
    }
    
    model_path = os.path.join(model_dir, "model_artifacts.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)
        
    print(f"\nModel artifacts saved successfully to {model_path}!")

if __name__ == "__main__":
    train_baseline_model()
