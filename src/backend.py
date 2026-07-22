"""
===============================================================================
CAREAGENT: FASTAPI BACKEND REST SERVER (backend.py)
===============================================================================
PURPOSE:
This module powers the high-performance FastAPI backend web service for CareAgent.
It exposes RESTful endpoints for patient registry queries, risk evaluation execution,
patient discharge simulation, interactive checklist persistence, clinical Q&A chatbot,
and monitoring bucket management.

KEY REST ENDPOINTS:
- GET  /patients                      : Queries patient registry with risk, SDOH, and care tier filters.
- GET  /patients/{patient_id}         : Fetches full profile, demographics, encounters, and latest evaluation.
- POST /patients/{patient_id}/discharge: Simulates a live discharge event & triggers CareAgent AI inference.
- POST /patients/{patient_id}/checklist: Updates completed status for clinical/SDOH checklist items.
- POST /chat                          : Interactive clinical Q&A assistant powered by Gemini 2.5 Flash.
- POST /patients/{patient_id}/monitor : Toggles a patient's starred/monitored status.
===============================================================================
"""

import os
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from src.agent import CareAgentOrchestrator

app = FastAPI(
    title="CareAgent Backend API",
    description="REST API for readmission risk prediction and care coordination recommendations.",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Orchestrator
# Will lazy-initialize tools and load data/models
orchestrator = CareAgentOrchestrator()

import threading

def run_pre_population():
    try:
        patients_df = orchestrator.data_tool.patients_df
        if patients_df is not None:
            populated_count = len(orchestrator.memory.memory)
            if populated_count < 20:
                print("Pre-populating patient memory evaluations...")
                p_ids = patients_df["patient_id"].head(100).tolist()
                for p_id in p_ids:
                    profile = orchestrator.data_tool.get_patient_profile(p_id)
                    if profile and profile.get("encounters"):
                        latest_enc = profile["encounters"][-1]
                        orchestrator.process_discharge_event(latest_enc["encounter_id"], bypass_gemini=True)
                print("Patient memory pre-population complete!")
    except Exception as e:
        print(f"Error pre-populating patient memory: {e}")

@app.on_event("startup")
def startup_event():
    # Start pre-population in a background thread to prevent blocking Uvicorn server startup
    thread = threading.Thread(target=run_pre_population)
    thread.start()

# Pydantic Schemas
class DischargeEvent(BaseModel):
    encounter_id: int

class ChatRequest(BaseModel):
    patient_id: int
    message: str

class ChatResponse(BaseModel):
    response: str

class ChecklistUpdateRequest(BaseModel):
    category: str  # "clinical" or "sdoh"
    index: int
    completed: bool

class MonitorUpdateRequest(BaseModel):
    monitored: bool


@app.get("/")
def read_root():
    return {"status": "healthy", "service": "CareAgent Backend API"}


@app.get("/patients")
def get_patients(
    risk_band: Optional[str] = Query(None, description="Filter by risk band (Low, Medium, High)"),
    care_level: Optional[str] = Query(None, description="Filter by care level (Routine, Enhanced, Intensive)"),
    sdoh_risk: Optional[str] = Query(None, description="Filter by SDOH risk level (Low, Moderate, High)"),
    diagnosis: Optional[str] = Query(None, description="Filter by latest diagnosis group"),
    search: Optional[str] = Query(None, description="Search by Patient ID"),
    monitored: Optional[bool] = Query(None, description="Filter by monitored status"),
    limit: int = 100,
    offset: int = 0
):
    try:
        # Load patients and their current evaluation status
        patients_df = orchestrator.data_tool.patients_df
        sdoh_df = orchestrator.data_tool.sdoh_df
        encounters_df = orchestrator.data_tool.encounters_df
        
        if patients_df is None or sdoh_df is None or encounters_df is None:
            raise HTTPException(status_code=500, detail="Data files not loaded. Please run data generator.")
            
        # Get latest encounter diagnosis group for each patient
        latest_encs = encounters_df.sort_values("admit_day").groupby("patient_id").last()[["diagnosis_group"]].reset_index()
        
        # Join patients with SDOH and latest encounters
        df = patients_df.merge(sdoh_df, on="patient_id", how="left")
        df = df.merge(latest_encs, on="patient_id", how="left")
        
        # Fetch all patient histories in a single batch read from Firestore to prevent 5000+ sequential queries
        db_docs = {}
        if orchestrator.memory.db:
            try:
                docs = orchestrator.memory.db.collection(orchestrator.memory.collection_name).get()
                db_docs = {doc.id: doc.to_dict() for doc in docs}
            except Exception as e:
                print(f"Error fetching batch documents from Firestore: {e}")
        
        results = []
        for _, row in df.iterrows():
            p_id = int(row["patient_id"])
            p_id_str = str(p_id)
            
            # Read from pre-fetched batch dict or fallback to local memory
            is_pat_monitored = False
            if p_id_str in db_docs:
                doc_data = db_docs[p_id_str]
                mem_runs = doc_data.get("history", [])
                is_pat_monitored = doc_data.get("monitored", False)
            elif not orchestrator.memory.db:
                mem_runs = orchestrator.memory.local_memory.get(p_id, [])
                is_pat_monitored = p_id in orchestrator.memory.local_monitored
            else:
                mem_runs = []
            
            if mem_runs:
                latest = mem_runs[-1]
                prob = latest["risk_results"]["readmit_probability"]
                band = latest["risk_results"]["readmit_risk_band"]
                care = latest["risk_results"]["care_management_level"]
            else:
                # Default/Initial fallback if not run yet
                # We do a fast heuristic prediction to populate the table
                sdoh_score = int(row["sdoh_score"])
                # Approximate probability
                prob = 0.05 + sdoh_score * 0.06
                band = "Low" if prob < 0.15 else ("Medium" if prob < 0.35 else "High")
                care = "Routine" if prob <= 0.20 and sdoh_score == 0 else ("Enhanced" if prob <= 0.40 or sdoh_score < 3 else "Intensive")
            
            # Apply filters
            if monitored is True and not is_pat_monitored:
                continue
            if risk_band and band != risk_band:
                continue
            if care_level and care != care_level:
                continue
            if sdoh_risk and row["sdoh_risk_level"] != sdoh_risk:
                continue
            if diagnosis and row["diagnosis_group"] != diagnosis:
                continue
            if search and str(p_id) != search.strip():
                continue
                
            results.append({
                "patient_id": p_id,
                "age": int(row["age"]),
                "sex": row["sex"],
                "insurance": row["insurance"],
                "language": row["language"],
                "sdoh_score": int(row["sdoh_score"]),
                "sdoh_risk_level": row["sdoh_risk_level"],
                "readmit_probability": prob,
                "readmit_risk_band": band,
                "care_management_level": care,
                "diagnosis_group": str(row["diagnosis_group"])
            })
            
        # Paginate results
        total = len(results)
        paginated_results = results[offset:offset+limit]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "patients": paginated_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}")
def get_patient_detail(patient_id: int):
    summary = orchestrator.get_patient_summary(patient_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    return summary

@app.post("/patients/{patient_id}/monitor")
def toggle_patient_monitoring(patient_id: int, request: MonitorUpdateRequest):
    success = orchestrator.memory.toggle_monitored(patient_id, request.monitored)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update monitoring status.")
    return {"status": "success", "monitored": request.monitored}



@app.post("/events/discharge")
def post_discharge_event(event: DischargeEvent):
    result = orchestrator.process_discharge_event(event.encounter_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/patients/{patient_id}/agent_summary")
def get_agent_summary(patient_id: int):
    summary = orchestrator.get_patient_summary(patient_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
        
    latest_eval = summary["latest_evaluation"]
    if "risk_results" not in latest_eval:
        # Latest eval might be the orchestrator default
        return {
            "readmit_probability": 0.05,
            "readmit_risk_band": "Low",
            "care_management_level": "Routine",
            "risk_drivers": [],
            "clinical_recommendations": ["No active encounters. Schedule standard wellness check-up."],
            "sdoh_interventions": [],
            "clinical_rationale": "No recent encounters found in patient record."
        }
        
    return {
        "readmit_probability": latest_eval["risk_results"]["readmit_probability"],
        "readmit_risk_band": latest_eval["risk_results"]["readmit_risk_band"],
        "care_management_level": latest_eval["risk_results"]["care_management_level"],
        "risk_drivers": latest_eval["recommendations"]["risk_drivers"],
        "clinical_recommendations": latest_eval["recommendations"]["clinical_recommendations"],
        "sdoh_interventions": latest_eval["recommendations"]["sdoh_interventions"],
        "clinical_rationale": latest_eval["recommendations"]["clinical_rationale"]
    }


@app.post("/patients/{patient_id}/checklist")
def update_checklist_item(patient_id: int, request: ChecklistUpdateRequest):
    success = orchestrator.memory.update_checklist(patient_id, request.category, request.index, request.completed)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update checklist item in memory.")
    return {"status": "success"}


@app.post("/chat", response_model=ChatResponse)
def chat_with_careagent(request: ChatRequest):
    patient_id = request.patient_id
    message = request.message.lower().strip()
    
    # 1. Fetch patient profile & risk assessment
    summary = orchestrator.get_patient_summary(patient_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
        
    profile = summary["profile"]
    latest_eval = summary["latest_evaluation"]
    
    # Extract risk & recommendations context
    risk_band = "Low"
    care_level = "Routine"
    prob = 0.05
    recs = {"risk_drivers": [], "clinical_recommendations": [], "sdoh_interventions": [], "clinical_rationale": ""}
    
    if "risk_results" in latest_eval:
        risk_band = latest_eval["risk_results"]["readmit_risk_band"]
        care_level = latest_eval["risk_results"]["care_management_level"]
        prob = latest_eval["risk_results"]["readmit_probability"]
        recs = latest_eval["recommendations"]
    elif "latest_evaluation" in latest_eval: # nested format from manual call
        risk_band = latest_eval["latest_evaluation"]["risk_results"]["readmit_risk_band"]
        care_level = latest_eval["latest_evaluation"]["risk_results"]["care_management_level"]
        prob = latest_eval["latest_evaluation"]["risk_results"]["readmit_probability"]
        recs = latest_eval["latest_evaluation"]["recommendations"]
        
    # Check if we should call Gemini API
    if orchestrator.recommendation_tool.client:
        try:
            client = orchestrator.recommendation_tool.client
            history_context = []
            for h in summary.get("history", [])[-5:]:
                history_context.append({
                    "timestamp": h["timestamp"],
                    "risk": h["risk_results"]["readmit_risk_band"],
                    "recs": h["recommendations"]["clinical_recommendations"]
                })
                
            # Clean recommendations list for text insertion
            clin_list = [r["text"] if isinstance(r, dict) else r for r in recs.get("clinical_recommendations", [])]
            sdoh_list = [r["text"] if isinstance(r, dict) else r for r in recs.get("sdoh_interventions", [])]
            
            prompt = f"""
You are CareAgent, the AI Care Coordinator. You are chatting with a healthcare provider about Patient {patient_id}.
Respond professionally, empathetically, and concisely in markdown.

Patient Context:
- Age: {profile['age']}, Sex: {profile['sex']}, Insurance: {profile['insurance']}
- SDOH Score: {profile.get('sdoh_score', 0)} ({profile.get('sdoh_risk_level', 'Low')} SDOH risk)
- Active SDOH Barriers: Housing={profile.get('housing_instability', 0)}, Food={profile.get('food_insecurity', 0)}, Transportation={profile.get('transportation_barrier', 0)}, Social Support={profile.get('low_social_support', 0)}
- Latest Diagnosis Group: {profile['encounters'][-1]['diagnosis_group'] if profile.get('encounters') else 'None'}
- Latest Readmission Risk: {prob:.1%} ({risk_band} risk band)
- Care Management Level: {care_level}

Current Recommendations:
- Clinical: {', '.join(clin_list)}
- SDOH Interventions: {', '.join(sdoh_list)}
- Rationale: {recs.get('clinical_rationale', '')}

Longitudinal History (Memory):
{json.dumps(history_context)}

User Query: "{request.message}"

Please address the user's query using the patient's data, risk models, and clinical background. Keep the response to 2-3 paragraphs max.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return ChatResponse(response=response.text)
        except Exception as e:
            print(f"Chat API Gemini call failed: {e}. Falling back to rule-based response.")
            
    # Intelligent Rules-Based Fallback Response
    resp_text = ""
    
    # 1. Ask about risk or why high risk
    if "why" in message or "risk" in message or "driver" in message or "factor" in message:
        resp_text = f"### CareAgent Risk Analysis\n"
        resp_text += f"Patient {patient_id} has a **{prob:.1%} readmission risk** (Risk Band: **{risk_band}**), requiring **{care_level} Care Management**.\n\n"
        resp_text += "**Key Risk Drivers Identified:**\n"
        if recs.get("risk_drivers"):
            for d in recs["risk_drivers"]:
                resp_text += f"- {d}\n"
        else:
            resp_text += f"- Active {profile['encounters'][-1]['diagnosis_group']} diagnosis.\n"
            if profile.get("sdoh_score", 0) > 0:
                resp_text += f"- Social Determinants of Health burden (Score: {profile['sdoh_score']}/4).\n"
        resp_text += f"\n**Rationale:** {recs.get('clinical_rationale', 'No explanation generated.')}"
        
    # 2. Ask about recommendations or plan
    elif "recommend" in message or "plan" in message or "do" in message or "action" in message or "next steps" in message:
        resp_text = f"### CareAgent Recommended Action Plan\n"
        resp_text += f"To mitigate the 30-day readmission risk for Patient {patient_id}, the following structured care plan is recommended:\n\n"
        
        resp_text += "**Clinical Interventions:**\n"
        if recs.get("clinical_recommendations"):
            for r in recs["clinical_recommendations"]:
                text = r["text"] if isinstance(r, dict) else r
                resp_text += f"- {text}\n"
        else:
            resp_text += "- Schedule standard post-discharge follow-up within 7-14 days.\n"
            
        if profile.get("sdoh_score", 0) > 0:
            resp_text += "\n**SDOH / Social Interventions:**\n"
            if recs.get("sdoh_interventions"):
                for r in recs["sdoh_interventions"]:
                    text = r["text"] if isinstance(r, dict) else r
                    resp_text += f"- {text}\n"
            else:
                resp_text += "- Screen for and coordinate social needs as appropriate.\n"
                
    # 3. Ask about SDOH or social or housing or food or transport
    elif any(kw in message for kw in ["sdoh", "social", "housing", "food", "transportation", "barrier", "support"]):
        resp_text = f"### Social Determinants of Health (SDOH) Summary\n"
        resp_text += f"Patient {patient_id} has an SDOH Risk Level of **{profile.get('sdoh_risk_level', 'Low')}** (Score: {profile.get('sdoh_score', 0)}/4).\n\n"
        resp_text += "**Active SDOH Flag Status:**\n"
        
        flags = [
            ("Housing Instability", "housing_instability", "Referral to housing navigator & emergency housing assistance."),
            ("Food Insecurity", "food_insecurity", "Enrollment in Medically Tailored Meals & food pantry list."),
            ("Transportation Barriers", "transportation_barrier", "Coordination of medical rideshare services."),
            ("Low Social Support", "low_social_support", "Deployment of Community Health Worker (CHW) check-ins.")
        ]
        
        any_flag = False
        for title, key, action in flags:
            status = "🔴 POSITIVE" if profile.get(key) == 1 else "🟢 Negative"
            resp_text += f"- **{title}**: {status}\n"
            if profile.get(key) == 1:
                any_flag = True
                resp_text += f"  - *Intervention:* {action}\n"
                
        if not any_flag:
            resp_text += "\nNo active SDOH barriers were flagged for this patient. Standard clinical care pathways apply."
            
    # 4. Discharge checks
    elif "discharge" in message or "before" in message or "discharge plan" in message:
        resp_text = f"### CareAgent Pre-Discharge Checklist\n"
        resp_text += f"For Patient {patient_id} ({care_level} Care Management), ensure the following are completed prior to discharge:\n\n"
        resp_text += "1. [ ] **Medication Reconciliation**: Complete and explain changes to patient/caregiver.\n"
        resp_text += f"2. [ ] **Follow-up Scheduled**: Confirm appointment within the timeframe ({'48 hours' if care_level=='Intensive' else '7 days' if care_level=='Enhanced' else '14 days'}).\n"
        if profile.get("transportation_barrier") == 1:
            resp_text += "3. [ ] **Transportation Secured**: Confirm rideshare or transit vouchers are arranged for the follow-up visit.\n"
        if profile.get("food_insecurity") == 1:
            resp_text += "4. [ ] **Nutritional Support**: Confirm the first delivery of Medically Tailored Meals is scheduled.\n"
        resp_text += "5. [ ] **Red Flags Review**: Hand patient the symptom warning signs and who to call if they worsen.\n"
        
    # Default greeting
    else:
        resp_text = f"Hello! I am **CareAgent**, your virtual clinical AI assistant. I've analyzed Patient {patient_id}'s EHR and SDOH data.\n\n"
        resp_text += f"**Patient Overview:**\n"
        resp_text += f"- **Age/Sex:** {profile['age']}-year-old {profile['sex']}\n"
        resp_text += f"- **Readmission Risk:** {risk_band} ({prob:.1%})\n"
        resp_text += f"- **Care Management Level:** {care_level}\n\n"
        resp_text += "You can ask me questions such as:\n"
        resp_text += "- *\"Why is this patient high risk?\"*\n"
        resp_text += "- *\"What recommendations should we follow?\"*\n"
        resp_text += "- *\"Detail the patient's SDOH status.\"*\n"
        resp_text += "- *\"What is the pre-discharge checklist?\"*"
        
    return ChatResponse(response=resp_text)
