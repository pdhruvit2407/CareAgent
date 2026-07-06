import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Optional Google GenAI SDK import
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class PatientDataTool:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.patients_df = None
        self.encounters_df = None
        self.sdoh_df = None
        self.load_data()
        
    def load_data(self):
        try:
            self.patients_df = pd.read_csv(os.path.join(self.data_dir, "careagent_patients_5000.csv"))
            self.encounters_df = pd.read_csv(os.path.join(self.data_dir, "careagent_encounters_5000.csv"))
            self.sdoh_df = pd.read_csv(os.path.join(self.data_dir, "careagent_sdoh_5000.csv"))
        except Exception as e:
            print(f"Warning: PatientDataTool failed to load data: {e}")
            
    def get_patient_profile(self, patient_id):
        # Refresh in case files were updated
        self.load_data()
        
        p_row = self.patients_df[self.patients_df["patient_id"] == patient_id]
        if p_row.empty:
            return None
            
        p_data = p_row.iloc[0].to_dict()
        
        # Add SDOH details
        sdoh_row = self.sdoh_df[self.sdoh_df["patient_id"] == patient_id]
        if not sdoh_row.empty:
            p_data.update(sdoh_row.iloc[0].to_dict())
            
        # Get encounters sorted by admit_day
        encs = self.encounters_df[self.encounters_df["patient_id"] == patient_id].sort_values("admit_day")
        p_data["encounters"] = encs.to_dict(orient="records")
        
        return p_data

    def get_encounter(self, encounter_id):
        self.load_data()
        enc_row = self.encounters_df[self.encounters_df["encounter_id"] == encounter_id]
        if enc_row.empty:
            return None
        return enc_row.iloc[0].to_dict()


class RiskModelTool:
    def __init__(self, model_path="src/model_artifacts.pkl"):
        self.model_path = model_path
        self.model = None
        self.encoders = None
        self.feature_cols = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.sdoh_flags = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                artifacts = pickle.load(f)
            self.model = artifacts["model"]
            self.encoders = artifacts["encoders"]
            self.feature_cols = artifacts["feature_cols"]
            self.categorical_cols = artifacts["categorical_cols"]
            self.numerical_cols = artifacts["numerical_cols"]
            self.sdoh_flags = artifacts["sdoh_flags"]
        else:
            print(f"Warning: Model artifacts not found at {self.model_path}. Risk score will use fallback rules.")

    def predict_risk(self, patient_profile, encounter):
        # Fallback if model is not loaded
        if self.model is None:
            # Simple rule-based score calculation
            score = 0.05
            if encounter.get("diagnosis_group") in ["CHF", "COPD", "Diabetes"]:
                score += 0.15
            score += patient_profile.get("sdoh_score", 0) * 0.08
            score += min(len(patient_profile.get("encounters", [])) * 0.05, 0.25)
            score = np.clip(score, 0.02, 0.85)
        else:
            # Prepare feature vector
            # Reconstruct variables as they were in encounters + patients join
            row_dict = {}
            # Demo / Patient fields
            for col in ["age", "sex", "insurance", "language", "housing_instability", 
                        "food_insecurity", "transportation_barrier", "low_social_support", 
                        "sdoh_score", "sdoh_risk_level"]:
                row_dict[col] = patient_profile.get(col)
                
            # Encounter fields
            row_dict["length_of_stay"] = encounter.get("length_of_stay")
            row_dict["encounter_type"] = encounter.get("encounter_type")
            row_dict["diagnosis_group"] = encounter.get("diagnosis_group")
            
            # Prior history fields computed dynamically up to this encounter
            encs = patient_profile.get("encounters", [])
            current_admit = encounter.get("admit_day")
            
            prior_encs = [e for e in encs if e["admit_day"] < current_admit]
            row_dict["prior_encounters"] = len(prior_encs)
            row_dict["prior_ed"] = sum(1 for e in prior_encs if e["encounter_type"] == "ED")
            row_dict["prior_inpatient"] = sum(1 for e in prior_encs if e["encounter_type"] == "Inpatient")
            
            # Create DataFrame row
            input_df = pd.DataFrame([row_dict])
            
            # Encode categorical variables
            for col in self.categorical_cols:
                le = self.encoders[col]
                val = str(input_df[col].iloc[0])
                # Handle unseen categories gracefully
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    # Fallback to the first class or default
                    input_df[col] = 0
            
            # Standardize columns ordering
            input_features = input_df[self.feature_cols]
            
            # Predict
            score = float(self.model.predict_proba(input_features)[0][1])
            
        # Determine risk band
        if score < 0.15:
            risk_band = "Low"
        elif score < 0.35:
            risk_band = "Medium"
        else:
            risk_band = "High"
            
        # Determine Care Management Level (Rule-based combining score + SDOH)
        sdoh_val = patient_profile.get("sdoh_score", 0)
        if score > 0.40 or (score > 0.25 and sdoh_val >= 3):
            care_level = "Intensive"
        elif score > 0.20 or sdoh_val >= 1:
            care_level = "Enhanced"
        else:
            care_level = "Routine"
            
        return {
            "readmit_probability": score,
            "readmit_risk_band": risk_band,
            "care_management_level": care_level
        }


class RecommendationTool:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client()
                print("Gemini client successfully initialized.")
            except Exception as e:
                print(f"Error initializing Gemini client: {e}. Falling back to rule-based engine.")

    def generate_recommendations(self, patient_profile, risk_analysis, memory_context=None):
        if self.client:
            try:
                prompt = self._build_prompt(patient_profile, risk_analysis, memory_context)
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                # Try parsing response as JSON
                text = response.text.strip()
                # Clean potential markdown wrappers
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                result = json.loads(text)
                return result
            except Exception as e:
                print(f"Gemini API execution failed: {e}. Falling back to rule-based generation.")
                
        return self._generate_rule_based(patient_profile, risk_analysis, memory_context)

    def _build_prompt(self, patient_profile, risk_analysis, memory_context):
        return f"""
You are CareAgent, an AI clinical coordinator. Your job is to generate highly personalized post-discharge recommendations and explain risk drivers for a patient.
You must output a raw JSON object and nothing else. No markdown blocks, no leading/trailing conversational text.

JSON Schema:
{{
  "risk_drivers": ["driver 1", "driver 2"],
  "clinical_recommendations": ["clinical plan 1", "clinical plan 2"],
  "sdoh_interventions": ["sdoh support 1", "sdoh support 2"],
  "clinical_rationale": "Detailed narrative explaining the clinical reasoning based on risk, age, history, and SDOH factors."
}}

Patient Profile:
- Patient ID: {patient_profile['patient_id']}
- Age: {patient_profile['age']} (Sex: {patient_profile['sex']})
- Insurance: {patient_profile['insurance']}
- Language: {patient_profile['language']}
- SDOH Flags:
  - Housing Instability: {patient_profile.get('housing_instability', 0)}
  - Food Insecurity: {patient_profile.get('food_insecurity', 0)}
  - Transportation Barrier: {patient_profile.get('transportation_barrier', 0)}
  - Low Social Support: {patient_profile.get('low_social_support', 0)}

Encounter History:
- Diagnosis Group: {patient_profile['encounters'][-1]['diagnosis_group']}
- Length of Stay: {patient_profile['encounters'][-1]['length_of_stay']} days
- Encounter Type: {patient_profile['encounters'][-1]['encounter_type']}
- Total Prior Encounters: {len(patient_profile['encounters']) - 1}

Risk Assessment:
- Predicted 30-Day Readmission Risk: {risk_analysis['readmit_probability']:.2%} (Band: {risk_analysis['readmit_risk_band']})
- Care Management Level: {risk_analysis['care_management_level']}

Memory/Longitudinal Context:
{json.dumps(memory_context) if memory_context else "No prior history in CareAgent memory."}

Provide concrete, actionable steps. Avoid generic medical advice. If they have SDOH flags, address them with specific resources (e.g. food delivery, housing navigators). Mention prior recommendations from memory and check if they need update or follow-up.
"""

    def _generate_rule_based(self, patient_profile, risk_analysis, memory_context):
        # Generate high-quality rule-based recommendations
        drivers = []
        clinical_recs = []
        sdoh_recs = []
        
        # Determine risk drivers
        prob = risk_analysis["readmit_probability"]
        diag = patient_profile["encounters"][-1]["diagnosis_group"]
        los = patient_profile["encounters"][-1]["length_of_stay"]
        
        if prob >= 0.35:
            drivers.append(f"Elevated baseline readmission probability ({prob:.1%})")
        if diag in ["CHF", "COPD", "Diabetes"]:
            drivers.append(f"Active chronic disease management for {diag}")
        if los > 5:
            drivers.append(f"Prolonged length of stay ({los} days) indicating clinical complexity")
            
        enc_count = len(patient_profile.get("encounters", [])) - 1
        if enc_count > 1:
            drivers.append(f"High utilization history ({enc_count} prior encounters in 12 months)")
            
        # Demographics and SDOH drivers
        sdoh_score = patient_profile.get("sdoh_score", 0)
        if sdoh_score > 0:
            drivers.append(f"Compounded SDOH burden (Score: {sdoh_score}/4)")
            
        # Generate Clinical Recs based on Care Level and diagnosis
        care_level = risk_analysis["care_management_level"]
        if care_level == "Intensive":
            clinical_recs.append("Schedule 48-hour post-discharge PCP / Specialist follow-up appointment.")
            clinical_recs.append("Initiate daily care manager telemonitoring check-ins for the first 14 days.")
            clinical_recs.append("Conduct comprehensive in-home pharmacist medication reconciliation within 3 days.")
        elif care_level == "Enhanced":
            clinical_recs.append("Schedule 7-day post-discharge PCP follow-up appointment.")
            clinical_recs.append("Schedule weekly care manager phone calls for the first 30 days.")
            clinical_recs.append("Conduct phone-based medication review with clinical pharmacist within 5 days.")
        else:
            clinical_recs.append("Schedule standard 14-day post-discharge PCP appointment.")
            clinical_recs.append("Provide patient with discharge self-care instructions and red flags checklist.")
            
        # Diagnosis specific clinical recommendations
        if diag == "CHF":
            clinical_recs.append("Enroll in CHF Care Pathway: daily weight logs, low-sodium diet counseling, and outpatient cardiology review.")
        elif diag == "COPD":
            clinical_recs.append("Confirm inhaler technique demonstration completed. Ensure oxygen supplies (if active) are delivered to home.")
        elif diag == "Diabetes":
            clinical_recs.append("Enroll in outpatient Diabetes self-management education. Review glucometer logs and insulin administration regimen.")
            
        # SDOH interventions
        if patient_profile.get("housing_instability") == 1:
            sdoh_recs.append("Refer to Medical-Legal Partnership & Housing Navigation services for emergency shelter/stable housing support.")
            drivers.append("Housing instability creates discharge placement risks")
        if patient_profile.get("food_insecurity") == 1:
            sdoh_recs.append("Enroll in Medically Tailored Meals (MTM) program and coordinate local food bank delivery.")
            drivers.append("Food insecurity compromises nutritional compliance and medication safety")
        if patient_profile.get("transportation_barrier") == 1:
            sdoh_recs.append("Coordinate medical transit rideshare services for all scheduled post-discharge appointments.")
            drivers.append("Transportation barrier raises clinic appointment no-show risks")
        if patient_profile.get("low_social_support") == 1:
            sdoh_recs.append("Deploy Community Health Worker (CHW) for twice-weekly home check-ins. Engage family/caregiver network.")
            drivers.append("Low social support limits adherence monitoring and emergency response")
            
        # Rationale narrative
        ins = patient_profile.get("insurance")
        age = patient_profile.get("age")
        rationale = f"Patient is a {age}-year-old with {ins} coverage discharged after a {los}-day {diag} encounter. "
        if prob >= 0.35:
            rationale += f"Due to High readmission risk ({prob:.1%}) and a care level of {care_level}, "
        else:
            rationale += f"Given a {risk_analysis['readmit_risk_band']} readmission risk ({prob:.1%}) and a care level of {care_level}, "
            
        if sdoh_score > 0:
            rationale += f"the care plan prioritizes addressing {sdoh_score} critical SDOH barriers. "
        else:
            rationale += "the care plan focuses primarily on standard clinical management. "
            
        if memory_context and len(memory_context) > 0:
            rationale += f"Longitudinal history indicates this is a recurring issue, with prior encounters requiring care plan adjustment."
            
        return {
            "risk_drivers": drivers,
            "clinical_recommendations": clinical_recs,
            "sdoh_interventions": sdoh_recs,
            "clinical_rationale": rationale
        }


class LoggingTool:
    def __init__(self, log_path="data/careagent_decisions_log.csv"):
        self.log_path = log_path
        
    def log_decision(self, patient_id, encounter_id, risk_results, recommendations):
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "readmit_probability": risk_results["readmit_probability"],
            "readmit_risk_band": risk_results["readmit_risk_band"],
            "care_management_level": risk_results["care_management_level"],
            "risk_drivers": json.dumps(recommendations["risk_drivers"]),
            "clinical_recs": json.dumps(recommendations["clinical_recommendations"]),
            "sdoh_recs": json.dumps(recommendations["sdoh_interventions"])
        }
        
        df_new = pd.DataFrame([log_entry])
        
        if os.path.exists(self.log_path):
            try:
                df_old = pd.read_csv(self.log_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.to_csv(self.log_path, index=False)
            except Exception:
                df_new.to_csv(self.log_path, index=False)
        else:
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            df_new.to_csv(self.log_path, index=False)
            
        print(f"Logged decision for Patient {patient_id}, Encounter {encounter_id} to {self.log_path}")


class PatientMemory:
    def __init__(self, memory_path="data/careagent_memory.pkl"):
        self.memory_path = memory_path
        self.memory = {}
        self.load_memory()
        
    def load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "rb") as f:
                    self.memory = pickle.load(f)
            except Exception:
                self.memory = {}
                
    def save_memory(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "wb") as f:
                pickle.dump(self.memory, f)
        except Exception as e:
            print(f"Error saving patient memory: {e}")
            
    def get_history(self, patient_id):
        return self.memory.get(patient_id, [])
        
    def update_history(self, patient_id, run_record):
        if patient_id not in self.memory:
            self.memory[patient_id] = []
        self.memory[patient_id].append(run_record)
        self.save_memory()

    def update_checklist(self, patient_id, category, index, completed):
        if patient_id in self.memory and self.memory[patient_id]:
            # Get latest run
            latest_run = self.memory[patient_id][-1]
            if category == "clinical":
                key = "clinical_recommendations"
            else:
                key = "sdoh_interventions"
                
            recs = latest_run.get("recommendations", {}).get(key, [])
            if 0 <= index < len(recs):
                if isinstance(recs[index], dict):
                    recs[index]["completed"] = completed
                else:
                    # If it's a string, convert to dict
                    recs[index] = {"text": recs[index], "completed": completed}
                self.save_memory()
                return True
        return False


class CareAgentOrchestrator:
    def __init__(self, data_dir="data", model_path="src/model_artifacts.pkl", memory_path="data/careagent_memory.pkl", log_path="data/careagent_decisions_log.csv"):
        self.data_tool = PatientDataTool(data_dir=data_dir)
        self.risk_tool = RiskModelTool(model_path=model_path)
        self.recommendation_tool = RecommendationTool()
        self.logging_tool = LoggingTool(log_path=log_path)
        self.memory = PatientMemory(memory_path=memory_path)
        
    def process_discharge_event(self, encounter_id):
        # 1. Fetch encounter
        enc = self.data_tool.get_encounter(encounter_id)
        if not enc:
            return {"error": f"Encounter ID {encounter_id} not found."}
            
        patient_id = enc["patient_id"]
        
        # 2. Fetch patient profile
        profile = self.data_tool.get_patient_profile(patient_id)
        if not profile:
            return {"error": f"Patient ID {patient_id} not found."}
            
        # 3. Fetch patient memory (longitudinal context)
        past_runs = self.memory.get_history(patient_id)
        memory_context = None
        if past_runs:
            memory_context = [
                {
                    "encounter_id": r["encounter_id"],
                    "timestamp": r["timestamp"],
                    "readmit_risk_band": r["risk_results"]["readmit_risk_band"],
                    "care_management_level": r["risk_results"]["care_management_level"],
                    "clinical_recommendations": r["recommendations"]["clinical_recommendations"],
                    "sdoh_interventions": r["recommendations"]["sdoh_interventions"]
                }
                for r in past_runs[-3:] # keep last 3 evaluations for context
            ]
            
        # 4. Predict risk
        risk_results = self.risk_tool.predict_risk(profile, enc)
        
        # 5. Generate recommendations
        recs = self.recommendation_tool.generate_recommendations(profile, risk_results, memory_context)
        
        # Format recommendations to include checklist completion state
        formatted_recs = {
            "risk_drivers": recs.get("risk_drivers", []),
            "clinical_recommendations": [{"text": r, "completed": False} if isinstance(r, str) else r for r in recs.get("clinical_recommendations", [])],
            "sdoh_interventions": [{"text": r, "completed": False} if isinstance(r, str) else r for r in recs.get("sdoh_interventions", [])],
            "clinical_rationale": recs.get("clinical_rationale", "")
        }
        
        # 6. Save to memory
        run_record = {
            "encounter_id": encounter_id,
            "timestamp": datetime.now().isoformat(),
            "risk_results": risk_results,
            "recommendations": formatted_recs
        }
        self.memory.update_history(patient_id, run_record)
        
        # 7. Log decision (serialize to logs using formatted strings)
        log_recs = {
            "risk_drivers": formatted_recs["risk_drivers"],
            "clinical_recommendations": [r["text"] for r in formatted_recs["clinical_recommendations"]],
            "sdoh_interventions": [r["text"] for r in formatted_recs["sdoh_interventions"]],
            "clinical_rationale": formatted_recs["clinical_rationale"]
        }
        self.logging_tool.log_decision(patient_id, encounter_id, risk_results, log_recs)
        
        return {
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "profile": {
                "age": profile["age"],
                "sex": profile["sex"],
                "insurance": profile["insurance"],
                "language": profile["language"],
                "sdoh_score": profile["sdoh_score"],
                "sdoh_risk_level": profile["sdoh_risk_level"]
            },
            "risk_results": risk_results,
            "recommendations": formatted_recs,
            "history_count": len(past_runs)
        }
        
    def get_patient_summary(self, patient_id):
        profile = self.data_tool.get_patient_profile(patient_id)
        if not profile:
            return {"error": f"Patient ID {patient_id} not found."}
            
        past_runs = self.memory.get_history(patient_id)
        
        # If no evaluations done yet, perform one on the latest encounter
        latest_run = None
        if past_runs:
            latest_run = past_runs[-1]
        else:
            encs = profile.get("encounters", [])
            if encs:
                latest_enc = encs[-1]
                latest_run = self.process_discharge_event(latest_enc["encounter_id"])
            else:
                latest_run = {
                    "risk_results": {
                        "readmit_probability": 0.05,
                        "readmit_risk_band": "Low",
                        "care_management_level": "Routine"
                    },
                    "recommendations": {
                        "risk_drivers": [],
                        "clinical_recommendations": [{"text": "No active encounters. Schedule standard wellness check-up.", "completed": False}],
                        "sdoh_interventions": [],
                        "clinical_rationale": "No recent encounters found in patient record."
                    }
                }
                
        return {
            "profile": profile,
            "latest_evaluation": latest_run,
            "history": past_runs
        }
