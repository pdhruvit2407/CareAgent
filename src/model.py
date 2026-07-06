import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def train_baseline_model(data_dir="data", model_dir="src"):
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
    sdoh_flags = ["housing_instability", "food_insecurity", "transportation_barrier", "low_social_support"]
    
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
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training set size: {len(X_train)} encounters")
    print(f"Testing set size: {len(X_test)} encounters")
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. {feature_cols[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # Save artifacts
    artifacts = {
        "model": rf_model,
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
