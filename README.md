# CareAgent: Clinical AI Agent for 30-Day Readmission Risk Mitigation & SDOH Coordination

CareAgent is an end-to-end, portfolio-grade healthcare data science and AI agent system. It integrates Electronic Health Record (EHR) data with Social Determinants of Health (SDOH) to predict 30-day hospital readmission risk, determine the required care management intensity, and generate actionable post-discharge recommendations for care teams.

The project features a synthetic generator that builds a realistic patient population, a baseline machine learning classifier trained on patient-level split data, a tool-using clinical AI agent layer with memory, a REST API backend, and a high-fidelity Streamlit dashboard.

---

## 1. Problem Context & SDOH Impact

30-day hospital readmissions represent a critical quality metric and financial challenge for healthcare systems globally. In the United States, the Hospital Readmissions Reduction Program (HRRP) penalizes hospitals with higher-than-expected readmission rates.

Traditional readmission risk models (like LACE or HOSPITAL scores) rely solely on clinical data. However, clinical factors only account for roughly 20% of health outcomes. The remaining 80% is driven by **Social Determinants of Health (SDOH)** and Health-Related Social Needs (HRSN). 

Patients facing food insecurity, housing instability, or transportation barriers are significantly less likely to fill prescriptions, attend follow-up appointments, or maintain therapeutic diets, leading to avoidable readmissions. CareAgent addresses this gap by combining clinical EHR data with SDOH flags to identify vulnerable patients and tailor clinical/social care plans before discharge.

---

## 2. System Architecture

CareAgent is built using a modern, decoupled healthcare AI architecture:

```
                  ┌───────────────────────────────┐
                  │      Streamlit Dashboard      │
                  │   (Registry, Details, Chat)   │
                  └───────────────┬───────────────┘
                                  │ (HTTP / JSON)
                                  ▼
                  ┌───────────────────────────────┐
                  │      FastAPI Backend API      │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                     CareAgent Orchestrator                       │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ PatientDataTool  │  │  RiskModelTool   │  │   Memory Tool  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘  │
│           │                     │                     │          │
│           ▼                     ▼                     ▼          │
│     [Data CSVs]           [ML Classifier]     [memory_store.pkl] │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      RecommendationTool                    │  │
│  │           (Gemini API / Detailed Clinical Rules)           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

- **Frontend Dashboard (Streamlit)**: Designed with a premium clinical aesthetic, providing registry filtering, demographic summaries, interactive timeline visualizations, care plan checklists, and an AI chat assistant.
- **Backend API (FastAPI)**: Serves endpoints for patient records, discharge event triggers, risk summary outputs, and conversational interactions.
- **Agent Orchestrator**: Coordinates specialized tools, manages persistent patient-specific memories, and handles decision logging.
- **Machine Learning Layer**: Predicts 30-day readmission risk using an ensemble random forest classifier trained on historical clinical and social features.

---

## 3. Dataset Specifications

CareAgent generates a synthetic dataset representing **5,000 patients** and **~12,500 clinical encounters** over a 365-day period, stored in three relational tables:

### 3.1 `patients` Table (`careagent_patients_5000.csv`)
Stores baseline demographic and SDOH profiles for 5,000 unique patients.
* **`patient_id`** (Primary Key): Integer index (1–5000)
* **`age`**: Integer (18–89, skewed older using a beta distribution to represent hospital-utilizing populations)
* **`sex`**: Categorical (`M`, `F`)
* **`insurance`**: Categorical (`Medicare`, `Medicaid`, `Commercial`, `Uninsured`), mapped programmatically based on age (e.g., Medicare dominant for age >= 65)
* **`language`**: Categorical (`English`, `Spanish`, `Other`)
* **SDOH/HRSN Flags** (Binary `0` or `1` representing patient barriers):
  * `housing_instability` (~20% positive rate)
  * `food_insecurity` (~25% positive rate)
  * `transportation_barrier` (~15% positive rate)
  * `low_social_support` (~20% positive rate)

### 3.2 `sdoh` Table (`careagent_sdoh_5000.csv`)
A patient-level summary of social barriers.
* **`patient_id`** (Foreign Key): Unique patient reference
* **`sdoh_score`**: Integer (0–4), representing the sum of active SDOH flags
* **`sdoh_risk_level`**: Categorical (`Low` for score 0, `Moderate` for score 1-2, `High` for score 3-4)

### 3.3 `encounters` Table (`careagent_encounters_5000.csv`)
Contains historical clinical events (1 to 5 per patient).
* **`encounter_id`** (Primary Key): Unique encounter reference (starting at 100001)
* **`patient_id`** (Foreign Key): Patient reference
* **`admit_day`**: Integer day of the year (1–365)
* **`discharge_day`**: `admit_day` + `length_of_stay`
* **`length_of_stay`**: Integer (1–10 days)
* **`encounter_type`**: Categorical (`Inpatient` 60%, `ED` 40%)
* **`diagnosis_group`**: Categorical (`CHF`, `COPD`, `Diabetes`, `General`)
* **`prior_encounters` / `prior_ed` / `prior_inpatient`**: Chronological cumulative metrics of prior patient visits
* **`readmit_30`** (Target): Binary label representing 30-day readmission, determined probabilistically using a weighted risk formula:
  $$\text{Prob} = 5\% + 12\% (\text{Chronic Diag}) + 6\% \times \text{SDOH Score} + 5\% \times \text{Prior Visits} + 3\% (\text{Inpatient})$$
  (Capped between 2% and 85%)

---

## 4. Modeling & Feature Engineering

### Feature Engineering
The modeling script joins `patients`, `sdoh`, and `encounters` on `patient_id` to create an encounter-level feature matrix. 

Categorical columns are encoded using label/ordinal encoders. Features include:
- **Demographics**: `age`, `sex`, `insurance`, `language`
- **Clinical**: `length_of_stay`, `encounter_type`, `diagnosis_group`, `prior_encounters`, `prior_ed`, `prior_inpatient`
- **SDOH**: `housing_instability`, `food_insecurity`, `transportation_barrier`, `low_social_support`, `sdoh_score`, `sdoh_risk_level`

### Patient-Level Validation Split
To prevent data leakage, the dataset is split at the **patient level** (80% train, 20% test). Splitting at the encounter level would lead to encounters of the same patient appearing in both train and test splits, causing over-optimistic test performance because static patient demographic and SDOH details would leak. 

### Model Performance
A `RandomForestClassifier` is trained to predict the binary target `readmit_30`. The model learns the probability of readmission, which is mapped to risk bands:
* **Low Risk**: $< 15\%$ probability
* **Medium Risk**: $15\%$ to $35\%$ probability
* **High Risk**: $\ge 35\%$ probability

Care management levels are assigned using rule-based boundaries on risk probability and SDOH scores:
* **Intensive Care Management**: Probability $> 40\%$ OR (Probability $> 25\%$ AND SDOH Score $\ge 3$)
* **Enhanced Care Management**: Probability $> 20\%$ OR SDOH Score $\ge 1$
* **Routine Care Management**: All other cases

---

## 5. Agent Orchestration & Memory

CareAgent utilizes a structured agent loop composed of specialized tools:

1. **PatientDataTool**: Fetches patient metadata, SDOH scores, and chronological encounters.
2. **RiskModelTool**: Performs inference using the trained Random Forest classifier.
3. **RecommendationTool**: Generates post-discharge care plans and explanations.
   * **Dual Mode**: If a `GEMINI_API_KEY` is present, it formats a comprehensive clinical prompt and calls the Gemini API to get a tailored clinical plan. If no key is set, it falls back to a deterministic clinical rule engine to generate highly specific checklists.
4. **LoggingTool**: Records predictions and recommended interventions to `data/careagent_decisions_log.csv`.
5. **PatientMemory**: Persists historical runs (`data/careagent_memory.pkl`) to retain longitudinal context (e.g., noting increasing visit frequencies) and prevent repeating duplicate interventions on consecutive encounters.

---

## 6. How to Run CareAgent

### Prerequisites
CareAgent requires Python 3.8 to 3.12. Ensure a virtual environment is used to isolate package dependencies.

### Step 1: Install Dependencies
Clone/navigate to the workspace and install the required packages:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Set API Key (Optional)
To run CareAgent with live Gemini LLM generation, set your API key in the shell environment:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
*Note: If no key is provided, CareAgent automatically operates in rule-based fallback mode, producing full features.*

### Step 3: Run Verification & Training
Run the built-in system verification tool to generate the synthetic data, train the baseline model, and verify the agent's pipelines:
```bash
python verify_system.py
```
This script runs a fast verification dataset, outputs model evaluation stats, and prints a mock agent run.

*To generate the full 5,000 patient dataset, run the generator script directly:*
```bash
python src/data_generator.py
```
*To train the model on the full generated dataset:*
```bash
python src/model.py
```

### Step 4: Launch Backend API
Start the FastAPI server:
```bash
uvicorn src.backend:app --reload --host 0.0.0.0 --port 8000
```
The API documentation will be available locally at [http://localhost:8000/docs](http://localhost:8000/docs).

### Step 5: Launch Streamlit Dashboard
In a separate terminal window (with the virtual environment activated), start the frontend dashboard:
```bash
streamlit run src/frontend.py
```
Open your browser to the local URL displayed (typically [http://localhost:8501](http://localhost:8501)).

---

## 7. Direct API Reference

CareAgent's FastAPI exposes the following REST endpoints:

* **`GET /patients`**: Returns a list of patients with their age, insurance, latest readmission risk, care management level, and SDOH status. Supports filtering parameters: `risk_band`, `care_level`, `sdoh_risk`, and `search` (by Patient ID).
* **`GET /patients/{patient_id}`**: Returns a patient's complete demographic profile, full clinical encounter history, and latest CareAgent evaluation.
* **`POST /events/discharge`**: Triggers CareAgent evaluation for a specific encounter ID. Body: `{"encounter_id": 100001}`.
* **`GET /patients/{patient_id}/agent_summary`**: Retrieves a detailed breakdown of the latest risk scores, risk drivers, clinical and social recommendations, and narrative clinical rationales.
* **`POST /chat`**: Enables conversational chat with CareAgent regarding a patient. Body: `{"patient_id": 123, "message": "Why is this patient at high risk?"}`.
