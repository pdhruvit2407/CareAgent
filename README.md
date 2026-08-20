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

---

## 6. Risk Timeline Logic & Probability Calculations

CareAgent simulates and predicts patient readmission risks using clinical, historical utilization, and social determinant factors. Below is the detailed mathematical framework and a concrete example mapping Patient ID 105.

### 6.1 Base Readmission Probability ($P_{\text{Base}}$)
First, a baseline readmission risk ($P_{\text{Base}}$) is calculated for each encounter:

$$P_{\text{Base}} = 0.05 + \Delta_{\text{Chronic}} + \Delta_{\text{SDOH}} + \Delta_{\text{History}} + \Delta_{\text{Type}}$$

Where:
* **Baseline Risk**: $0.05$ (5% starting risk for any hospitalization)
* **$\Delta_{\text{Chronic}}$ (Chronic Disease Adjustment)**: $+0.12$ if primary diagnosis is CHF, COPD, Diabetes, Asthma, or Hypertension. Otherwise $+0.00$.
* **$\Delta_{\text{SDOH}}$ (Social Barriers Influence)**: $+0.04 \times (\text{SDOH Score})$ (Adds up to $+0.24$ if all 6 screening flags are positive).
* **$\Delta_{\text{History}}$ (Utilization History)**: $+0.05 \times (\text{Prior Encounters})$ (Capped at a maximum influence of $+0.25$).
* **$\Delta_{\text{Type}}$ (Encounter Severity)**: $+0.03$ if the encounter type is `"Inpatient"`. Otherwise $+0.00$.

### 6.2 Independent Disjoint Risk Windows
Because the three risk timelines represent independent, disjoint 30-day windows (Days 1–30, Days 31–60, and Days 61–90), the probabilities are scaled down over time to reflect post-acute stabilization:

* **30-Day Risk (Days 1–30)**:
  $$P_{30} = \text{clip}(P_{\text{Base}}, 0.02, 0.85)$$
* **60-Day Risk (Days 31–60)**:
  $$P_{60} = \text{clip}(P_{\text{Base}} \times 0.5, 0.01, 0.40)$$
* **90-Day Risk (Days 61–90)**:
  $$P_{90} = \text{clip}(P_{\text{Base}} \times 0.3, 0.01, 0.25)$$

*Note: In the synthetic database generator, outcomes (`readmit_30`, `readmit_60`, `readmit_90`) are sampled via a Bernoulli trial ($0$ or $1$) using these probabilities.*

---

### 6.3 Concrete Calculation Example (Patient ID 105)

#### Patient Profile Details:
* **Patient ID**: 105
* **Demographics**: 74-year-old female
* **Encounter Diagnosis**: CHF (Congestive Heart Failure)
* **Encounter Type**: Inpatient
* **Prior encounters (past 12 months)**: 2
* **Positive SDOH Flags**: 3 (Food Insecurity, Income Strain, Transportation Barrier)

#### Step-by-Step Calculation:
1. **$P_{\text{Base}}$ Calculation**:
   * Base: $0.05$
   * $\Delta_{\text{Chronic}}$ (CHF): $+0.12$
   * $\Delta_{\text{SDOH}}$ (3 flags): $3 \times 0.04 = +0.12$
   * $\Delta_{\text{History}}$ (2 prior): $2 \times 0.05 = +0.10$
   * $\Delta_{\text{Type}}$ (Inpatient): $+0.03$
   * **Total $P_{\text{Base}}$**: $0.05 + 0.12 + 0.12 + 0.10 + 0.03 = \mathbf{0.42}$ (42%)

2. **Window Risks & Threshold Categorization**:
   * **30-Day Risk**: $\text{clip}(0.42, 0.02, 0.85) = \mathbf{42.0\%}$ $\rightarrow$ **High Risk** (since $\ge 35\%$)
   * **60-Day Risk**: $\text{clip}(0.42 \times 0.5, 0.01, 0.40) = \mathbf{21.0\%}$ $\rightarrow$ **Medium Risk** (since $\ge 15\%$ and $< 35\%$)
   * **90-Day Risk**: $\text{clip}(0.42 \times 0.3, 0.01, 0.25) = \mathbf{12.6\%}$ $\rightarrow$ **Low Risk** (since $< 15\%$)

