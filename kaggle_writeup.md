# Kaggle Portfolio Project Write-Up: CareAgent
## Clinical AI Agent System for 30-Day Hospital Readmission Risk Mitigation & SDOH Coordination

**Author:** Kaggle Vibe Coder & AI Health Data Scientist  
**Project Repository:** [CareAgent Codebase](file:///Users/w01334/Documents/AI Project/CareAgent)  
**Live Frontend Dashboard:** [CareAgent Streamlit App](https://careagent-frontend-1038052745096.us-west1.run.app)  
**Live API Documentation:** [FastAPI Backend Service](https://careagent-backend-1038052745096.us-west1.run.app/docs)  

---

## 1. Executive Summary

**CareAgent** is an end-to-end, clinical-grade AI agent system designed to predict 30-day hospital readmission risk and automate social care coordination. 

While traditional clinical prediction models (e.g., LACE or HOSPITAL scores) operate exclusively on electronic health record (EHR) data, they neglect **Social Determinants of Health (SDOH)**, which drive up to 80% of health outcomes. CareAgent bridges this gap by integrating clinical data (length of stay, diagnoses, prior utilization) with social risk factors (food insecurity, housing instability, transportation barriers, social support) to:
1. **Predict** 30-day readmission risk using an ensemble machine learning classifier.
2. **Assign** patients to dynamic Care Management Levels (Routine, Enhanced, Intensive).
3. **Generate** highly actionable clinical and social care recommendations via a tool-using AI Agent equipped with persistent longitudinal memory.
4. **Present** these insights through a sleek, responsive clinical dashboard containing a patient registry, timeline visuals, persistent care checklists, and an interactive conversational assistant.

---

## 2. The Clinical Challenge: Hospital Readmissions & SDOH

### 2.1 The Multi-Billion Dollar Problem
Unplanned 30-day hospital readmissions are a key indicator of healthcare fragmentation and high costs. In the United States, the Centers for Medicare & Medicaid Services (CMS) penalizes hospitals with excessive readmissions under the **Hospital Readmissions Reduction Program (HRRP)**. 

### 2.2 The Role of Social Determinants of Health (SDOH)
A patient discharged after a congestive heart failure (CHF) exacerbation might receive excellent inpatient care, but if they return to a home with **housing instability**, face **food insecurity** that makes a low-sodium diet impossible, or have **transportation barriers** preventing them from attending a 7-day post-discharge check-up, they will likely be readmitted. 

CareAgent addresses this clinical challenge by building a unified risk engine that treats social needs as first-class citizens alongside clinical variables.

---

## 3. Synthetic Data Engineering Methodology

To train and validate the CareAgent system, we developed a highly customizable, relational synthetic data generator containing **5,000 unique patients** and **12,041 clinical encounters** representing a synthetic year.

```
       ┌────────────────────────┐         ┌────────────────────────┐
       │     patients table     │         │       sdoh table       │
       │     (5,000 records)    │         │     (5,000 records)    │
       └───────────┬────────────┘         └───────────┬────────────┘
                   │                                  │
                   │ (1-to-1 Join)                    │ (1-to-1 Join)
                   ▼                                  ▼
       ┌───────────────────────────────────────────────────────────┐
       │                      encounters table                     │
       │              (12,041 records, 1-to-5 per patient)         │
       └───────────────────────────────────────────────────────────┘
```

### 3.1 Patient Registry Demographics (`patients` table)
* **Age Distribution**: Generates patients aged 18 to 89. We utilized a Beta distribution ($\alpha=3, \beta=2$) to skew the population older, representing a realistic hospital-utilizing cohort.
* **Linguistic Diversity**: Mapped Preferred Languages (80% English, 15% Spanish, 5% Other).
* **Age-Dependent Insurance**: Mapped insurance coverage dynamically:
  * For patients $\ge 65$: 88% Medicare, 7% Medicaid, 3% Commercial, 2% Uninsured.
  * For patients $< 65$: 50% Commercial, 30% Medicaid, 5% Medicare, 15% Uninsured.
* **SDOH Risk Flags**: Binary flags mapping specific social needs based on target positive rates:
  * `housing_instability` (~20% positive rate)
  * `food_insecurity` (~25% positive rate)
  * `transportation_barrier` (~15% positive rate)
  * `low_social_support` (~20% positive rate)

### 3.2 Social Summary Profile (`sdoh` table)
Summarizes patient barriers by defining:
* `sdoh_score`: Sum of the 4 binary SDOH flags (0–4).
* `sdoh_risk_level`: Categorical mapping based on score: `Low` (score 0), `Moderate` (score 1–2), `High` (score 3–4).

### 3.3 Clinical Encounters (`encounters` table)
Each patient receives 1 to 5 encounters chronologically spaced over a 365-day year:
* **Diagnosis Groups**: Chronic disease conditions CHF (20%), COPD (20%), Diabetes (25%), and General (35%).
* **Utilization History**: Dynamically calculates cumulative prior visits, separating prior Emergency Department (ED) visits from prior Inpatient stays.
* **Probabilistic Readmission Label (`readmit_30`)**: Mapped using a noisy, weighted sigmoid-like Bernoulli trial:
  $$P(\text{Readmit}) = \text{clip}\left(0.05 + 0.12(\text{Chronic Diag}) + 0.06(\text{SDOH Score}) + \min(0.05 \times \text{Prior Visits}, 0.25) + 0.03(\text{Inpatient}), 0.02, 0.85\right)$$

---

## 4. Predictive Modeling & Evaluation

### 4.1 Feature Engineering
The three tables are merged on `patient_id`. Categorical variables (`sex`, `insurance`, `language`, `encounter_type`, `diagnosis_group`, and `sdoh_risk_level`) are mapped using scikit-learn Label Encoders. 

### 4.2 Grouped Patient-Level Validation Split
> [!IMPORTANT]
> **To prevent data leakage, the train/test split (80/20) was performed at the patient-level rather than the encounter-level.**
> If we split randomly at the encounter level, multiple encounters for the same patient would reside in both the training and testing sets. Because static demographic features (like `insurance`) and SDOH flags are identical across encounters, the model would memorize patient-specific characteristics, leading to artificial, over-optimistic test performance. Splitting by patient ID guarantees that the test set evaluates patients that the model has never seen.

### 4.3 Model Performance
An ensemble `RandomForestClassifier` (100 estimators, max depth of 8) was trained:
* **Overall Test Accuracy**: **75.69%**
* **ROC-AUC Score**: **0.6324**

### 4.4 Feature Importances
The model's feature importance ranking shows that clinical utilization and SDOH play a dominant role in predicting readmissions:
1. **Age**: 17.47% (Older patients present higher clinical vulnerability)
2. **Diagnosis Group**: 12.64% (Chronic conditions CHF/COPD/Diabetes have high readmit baselines)
3. **Length of Stay**: 10.72% (Longer inpatient stays map to higher severity)
4. **Prior Encounters**: 10.72% (Frequent fliers have higher return probability)
5. **SDOH Score**: 7.71% (Total count of active social barriers)

---

## 5. Stateful Agent Architecture

CareAgent is implemented as a tool-using AI agent orchestrated in Python:

```
┌────────────────────────────────────────────────────────┐
│               CareAgent Orchestrator                   │
├────────────────────────────────────────────────────────┤
│  1. PatientDataTool  -->  Loads Patients, EHR & SDOH   │
│  2. RiskModelTool    -->  Runs Classifier Inference    │
│  3. LoggingTool      -->  Saves Decisions to CSV       │
│  4. PatientMemory    -->  Loads & Saves memory.pkl     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  RecommendationTool                    │
│    (Checks Memory for Context & Deduplication)         │
├────────────────────────────────────────────────────────┤
│  ▲ Live GEMINI_API_KEY      ▼ No API Key (Fallback)    │
│  Generate tailored clinical  Generate detailed clinical│
│  JSON via Gemini 2.5 Flash   and SDOH rule checklists  │
└────────────────────────────────────────────────────────┘
```

### 5.1 Orchestration Tools
* **`PatientDataTool`**: Handles querying of patient details, demographics, and encounter histories.
* **`RiskModelTool`**: Performs inference using the saved Random Forest model, assigning risk probabilities and bands.
* **`LoggingTool`**: Appends audit logs for risk outputs and decisions to `data/careagent_decisions_log.csv`.
* **`RecommendationTool`**: Dual-mode engine. If a `GEMINI_API_KEY` is present, it constructs a rich clinical prompt with history, memory, and patient characteristics, generating highly tailored plans. If the key is missing, it falls back to a deterministic, clinical rule-engine that outputs comprehensive checklists.
* **`PatientMemory`**: Persists evaluation histories to `data/careagent_memory.pkl` to retain context (e.g. "patient has had 3 ED visits in 6 months") and prevent duplicate interventions.

---

## 6. Decoupled Cloud Architecture

CareAgent was deployed to **Google Cloud Run** using a fully decoupled containerized layout:

### 6.1 Backend API Service (FastAPI)
Exposes endpoints to retrieve patient records, trigger discharge evaluations, fetch agent summaries, and perform chat interactions. 
* **Dynamic Binding**: Modified the FastAPI entrypoint to bind dynamically to the `$PORT` environment variable (typically `8080` in Cloud Run) to pass Google's health checks.
* **Endpoint Summary**:
  * `GET /patients`: Filtered patient lists.
  * `GET /patients/{id}`: Detailed demographics and timeline.
  * `POST /events/discharge`: Triggers agent execution for a new discharge event.
  * `POST /patients/{id}/checklist`: Persists checklist completions directly in `PatientMemory`.
  * `POST /chat`: Multi-turn conversational chat with the agent regarding the patient.

### 6.2 Frontend Dashboard Service (Streamlit)
Streamlit is deployed as a separate Cloud Run service. It communicates with the backend via the `BACKEND_URL` environment variable.
* **Persistent Checklists**: Binds Streamlit's checkbox widget state to the backend `POST /checklist` endpoint. Clicking a checklist item immediately saves the task status to the database, ensuring selections persist even when switching between patients.

---

## 7. Key Findings & Future Directions

### 7.1 Insights
* **The SDOH Multiplier**: The addition of SDOH features significantly improved the model's ability to differentiate risk in patients with identical diagnoses, showing that social risk acts as a key multiplier.
* **Validation Strictness**: Patient-level validation is essential. Encounter-level splits lead to data leakage and artificial scores that fail when deployed to production.

### 7.2 Future Scope
* **Live EHR Integration**: Transition from synthetic CSV files to FHIR API endpoints (e.g., Epic or Cerner).
* **Distributed Memory**: Move `careagent_memory.pkl` from local disk storage to a managed database (e.g., Firestore or Cloud SQL) to allow multi-instance horizontal scaling in production.
* **LLM Guardrails**: Implement guardrail layers (e.g., NeMo Guardrails or Vertex AI Safety Filters) to ensure clinical chats remain within strict medical boundaries.
