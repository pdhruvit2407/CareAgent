# CareAgent: Clinical AI Agent for Multi-Horizon Readmission Risk Timeline & SDOH Coordination

CareAgent is an end-to-end clinical decision support platform and workflow assistant. It integrates Electronic Health Record (EHR) data with Social Determinants of Health (SDOH) to predict multi-horizon readmission risk, determine the required care management intensity, and generate actionable post-discharge care checklists.

The project features a synthetic generator that builds a balanced clinical cohort, three independent machine learning classifiers, a stateful tool-using orchestrator with Firestore memory, a FastAPI backend, and a Streamlit dashboard.

---

## 1. Problem Context & SDOH Impact

30-day, 60-day, and 90-day readmissions represent a critical quality metric and financial challenge for healthcare systems. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face substantial penalties for excess readmissions.

CareAgent addresses this challenge by combining clinical EHR details with **6 Social Determinants of Health (SDOH)**. Because non-clinical social factors account for roughly 80% of health outcomes, incorporating these barriers enables the AI to generate targeted care plans (e.g., medically tailored meals, co-pay assistance, transportation rideshares) to prevent avoidable returns.

---

## 2. Key Features

* **Balanced Diagnoses (5 Groups)**: Covers Congestive Heart Failure (CHF), Chronic Obstructive Pulmonary Disease (COPD), Diabetes, Asthma, and Hypertension.
* **6 SDOH Barriers**: Screens for Food Insecurity, Income Strain, Housing Instability, Education & Literacy, Social Isolation, and Transportation Barriers.
* **Independent Disjoint Risk Timeline**: Uses three independent Random Forest Classifiers to calculate risk across disjoint time intervals:
  - **30-Day Risk**: Probability of readmission during Days 1–30.
  - **60-Day Risk**: Probability of readmission during Days 31–60.
  - **90-Day Risk**: Probability of readmission during Days 61–90.
* **History-Wide Care Plans**: Analyzes a patient's entire encounter history to compile clinical care recommendations for **every** unique disease diagnosed in their past (e.g., managing both CHF and Hypertension).
* **Stateful Firestore Integration**: Persists care plans, checklists, and agent chat transcripts using Google Cloud Firestore (Native Mode).

---

## 3. Directory Structure

```
CareAgent/
├── data/
│   ├── careagent_patients_5000.csv     # Demographics database
│   ├── careagent_sdoh_5000.csv         # 6-category SDOH surveys
│   └── careagent_encounters_5000.csv   # Historical encounters
├── src/
│   ├── data_generator.py               # Synthetic cohort generator
│   ├── model.py                        # Trains the 3 Random Forest models
│   ├── model_artifacts.pkl             # Serialized models and encoders
│   ├── agent.py                        # RiskModelTool and Gemini RAG Prompt Orchestrator
│   ├── backend.py                      # FastAPI REST API endpoints
│   └── frontend.py                     # Streamlit dashboard UI
├── requirements.txt                    # Project dependencies
├── Dockerfile.backend                  # Backend container configuration
└── Dockerfile.frontend                 # Frontend container configuration
```

---

## 4. Modeling & Performance

The modeling script joins demographic, SDOH, and encounter records. Features are split at the patient level to prevent data leakage. CareAgent trains **three separate Random Forest Classifiers** on the independent disjoint targets:
* **Days 1–30 Model**: Accuracy: **72.09%**, ROC-AUC: **0.5823**
* **Days 31–60 Model**: ROC-AUC: **0.5480**
* **Days 61–90 Model**: ROC-AUC: **0.5653**

Care management levels are routed strictly by the **30-day primary risk** and aggregate SDOH scores to match standard hospital billing codes.

---

## 5. How to Run CareAgent Locally

### Step 1: Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Generate Cohort & Train Models
```bash
# Generate the full dataset
python src/data_generator.py

# Train the independent models
python src/model.py
```

### Step 3: Run FastAPI Backend
```bash
uvicorn src.backend:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Run Streamlit Frontend
In a new terminal window:
```bash
export BACKEND_URL="http://localhost:8000"
streamlit run src/frontend.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
