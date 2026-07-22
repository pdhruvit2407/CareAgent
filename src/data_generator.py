"""
===============================================================================
CAREAGENT: SYNTHETIC HEALTHCARE DATA GENERATOR (data_generator.py)
===============================================================================
PURPOSE:
This script generates a realistic, clinically representative synthetic dataset of
5,000 patients and ~12,000 hospital encounters. It simulates electronic health record
(EHR) demographic fields, Social Determinants of Health (SDOH) screening surveys, 
and 30-day hospital readmission targets driven by realistic clinical risk equations.

KEY OUTPUT DATASETS GENERATED (saved to data/):
1. careagent_patients_5000.csv   : Patient demographics, insurance, & SDOH survey flags.
2. careagent_sdoh_5000.csv       : Computed SDOH risk scores (0-4) and risk tiers.
3. careagent_encounters_5000.csv : Longitudinal hospital admission/discharge records.
===============================================================================
"""

import os
import pandas as pd
import numpy as np

def generate_synthetic_data(data_dir="data", num_patients=5000):
    """
    Generates synthetic patient demographics, SDOH screening scores, and encounter records.
    
    Parameters:
        data_dir (str): Directory where output CSV files will be saved.
        num_patients (int): Number of synthetic patient records to generate.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Generating synthetic healthcare data for {num_patients} patients...")
    
    # 1. Patients Table
    patient_ids = np.arange(1, num_patients + 1)
    
    # Demographics
    # Skew age older: beta distribution scaled between 18 and 89
    # a=3, b=2 yields a distribution skewed towards the higher end (mean = 3/(3+2) = 0.6)
    ages = (18 + 71 * np.random.beta(a=3, b=2, size=num_patients)).astype(int)
    sexes = np.random.choice(["M", "F"], size=num_patients, p=[0.48, 0.52])
    languages = np.random.choice(["English", "Spanish", "Other"], size=num_patients, p=[0.80, 0.15, 0.05])
    
    # Insurance dependent on age
    insurances = []
    for age in ages:
        if age >= 65:
            insurances.append(np.random.choice(["Medicare", "Medicaid", "Commercial", "Uninsured"], p=[0.88, 0.07, 0.03, 0.02]))
        else:
            insurances.append(np.random.choice(["Medicare", "Medicaid", "Commercial", "Uninsured"], p=[0.05, 0.30, 0.50, 0.15]))
            
    # SDOH / HRSN binary flags (revised positive rates)
    food_insecurity = np.random.binomial(1, 0.24, size=num_patients)
    income_barrier = np.random.binomial(1, 0.30, size=num_patients)
    housing_instability = np.random.binomial(1, 0.10, size=num_patients)
    education_literacy_barrier = np.random.binomial(1, 0.23, size=num_patients)
    low_social_support = np.random.binomial(1, 0.06, size=num_patients)
    transportation_barrier = np.random.binomial(1, 0.07, size=num_patients)
    
    patients_df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "sex": sexes,
        "insurance": insurances,
        "language": languages,
        "food_insecurity": food_insecurity,
        "income_barrier": income_barrier,
        "housing_instability": housing_instability,
        "education_literacy_barrier": education_literacy_barrier,
        "low_social_support": low_social_support,
        "transportation_barrier": transportation_barrier
    })
    
    # 2. SDOH Table
    sdoh_scores = food_insecurity + income_barrier + housing_instability + education_literacy_barrier + low_social_support + transportation_barrier
    sdoh_risk_levels = []
    for score in sdoh_scores:
        if score == 0:
            sdoh_risk_levels.append("Low")
        elif score <= 3:
            sdoh_risk_levels.append("Moderate")
        else:
            sdoh_risk_levels.append("High")
            
    sdoh_df = pd.DataFrame({
        "patient_id": patient_ids,
        "sdoh_score": sdoh_scores,
        "sdoh_risk_level": sdoh_risk_levels
    })
    
    # 3. Encounters Table
    encounters_list = []
    encounter_id_counter = 100001
    
    # Diagnosis groups (Specific Chronic Disease Cohorts)
    diag_groups = ["CHF", "COPD", "Diabetes", "Asthma", "Hypertension"]
    diag_probs = [0.20, 0.20, 0.20, 0.20, 0.20]
    
    for i in range(num_patients):
        p_id = patient_ids[i]
        p_age = ages[i]
        sdoh_score = sdoh_scores[i]
        
        # Determine number of encounters for this patient (1 to 5)
        # Shift probability to higher count if older or higher SDOH score
        num_enc_prob = [0.4, 0.3, 0.15, 0.10, 0.05]
        if p_age >= 65 or sdoh_score >= 2:
            num_enc_prob = [0.2, 0.3, 0.25, 0.15, 0.10]
            
        num_encounters = np.random.choice([1, 2, 3, 4, 5], p=num_enc_prob)
        
        current_day = np.random.randint(1, 60)
        
        for enc_idx in range(num_encounters):
            # Calculate prior counts up to this encounter
            prior_encounters = enc_idx
            # Simulating prior encounter types (approx 40% ED, 60% Inpatient)
            prior_ed = sum(1 for e in encounters_list if e["patient_id"] == p_id and e["encounter_type"] == "ED")
            prior_inpatient = sum(1 for e in encounters_list if e["patient_id"] == p_id and e["encounter_type"] == "Inpatient")
            
            los = np.random.randint(1, 11)
            admit_day = int(current_day)
            discharge_day = admit_day + los
            
            enc_type = np.random.choice(["Inpatient", "ED"], p=[0.60, 0.40])
            diag = np.random.choice(diag_groups, p=diag_probs)
            
            # Readmission probability formula
            # Base risk: 5%
            # Chronic diagnosis: CHF/COPD/Diabetes/Asthma/Hypertension (+12%)
            # SDOH score: +4% per flag (up to 24% for 6 flags)
            # Prior encounter count: +5% per prior encounter (capped influence)
            # Inpatient encounter has slightly higher risk of readmission: +3%
            p_readmit = 0.05
            if diag in ["CHF", "COPD", "Diabetes", "Asthma", "Hypertension"]:
                p_readmit += 0.12
            p_readmit += sdoh_score * 0.04
            p_readmit += min(prior_encounters * 0.05, 0.25)
            if enc_type == "Inpatient":
                p_readmit += 0.03
                
            # Cap readmission probability
            p_readmit = np.clip(p_readmit, 0.02, 0.85)
            
            readmit_30 = np.random.binomial(1, p_readmit)
            
            encounters_list.append({
                "encounter_id": encounter_id_counter,
                "patient_id": p_id,
                "admit_day": admit_day,
                "discharge_day": discharge_day,
                "length_of_stay": los,
                "encounter_type": enc_type,
                "diagnosis_group": diag,
                "readmit_30": readmit_30,
                # Store these in raw encounters for ease of modeling, though we'll re-calculate chronologically in model
                "prior_encounters": prior_encounters,
                "prior_ed": prior_ed,
                "prior_inpatient": prior_inpatient
            })
            
            encounter_id_counter += 1
            # Advance day for next encounter
            current_day = discharge_day + np.random.randint(10, 80)
            if current_day > 365:
                break # Don't generate encounters past 1 year
                
    encounters_df = pd.DataFrame(encounters_list)
    
    # Save CSVs
    patients_df.to_csv(os.path.join(data_dir, "careagent_patients_5000.csv"), index=False)
    sdoh_df.to_csv(os.path.join(data_dir, "careagent_sdoh_5000.csv"), index=False)
    encounters_df.to_csv(os.path.join(data_dir, "careagent_encounters_5000.csv"), index=False)
    
    print(f"Data generated successfully!")
    print(f"Patients: {len(patients_df)} rows")
    print(f"SDOH: {len(sdoh_df)} rows")
    print(f"Encounters: {len(encounters_df)} rows")
    
if __name__ == "__main__":
    generate_synthetic_data()
