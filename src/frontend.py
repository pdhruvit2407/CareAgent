import streamlit as st
import pandas as pd
import requests
import json

# Set Page Config
st.set_page_config(
    page_title="CareAgent Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
# Backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Inject Custom CSS for Premium Clinical Aesthetics (Outfit font, Glassmorphism, Gradients)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main Fonts */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Custom header banner */
.header-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: white;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 1.1rem;
    font-weight: 300;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* Glassmorphic cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

/* Gradient metric cards */
.metric-card {
    background: linear-gradient(135deg, #111e38 0%, #1a2a4a 100%);
    padding: 1.25rem;
    border-radius: 12px;
    border-left: 5px solid #00c6ff;
    box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.15);
    margin-bottom: 1rem;
}
.metric-card.critical {
    border-left-color: #ff4b4b;
}
.metric-card.warning {
    border-left-color: #ffa500;
}
.metric-card.success {
    border-left-color: #00e676;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.metric-lbl {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.7;
    margin: 0;
}

/* Custom Badges */
.badge {
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    text-align: center;
}
.badge-high {
    background-color: rgba(255, 75, 75, 0.15);
    color: #ff4b4b;
    border: 1px solid rgba(255, 75, 75, 0.3);
}
.badge-medium {
    background-color: rgba(255, 165, 0, 0.15);
    color: #ffa500;
    border: 1px solid rgba(255, 165, 0, 0.3);
}
.badge-low {
    background-color: rgba(0, 230, 118, 0.15);
    color: #00e676;
    border: 1px solid rgba(0, 230, 118, 0.3);
}
.badge-intensive {
    background: linear-gradient(135deg, rgba(239, 83, 80, 0.2) 0%, rgba(229, 57, 53, 0.2) 100%);
    color: #ef5350;
    border: 1px solid #ef5350;
    box-shadow: 0 0 10px rgba(239, 83, 80, 0.15);
}
.badge-enhanced {
    background: linear-gradient(135deg, rgba(255, 167, 38, 0.2) 0%, rgba(245, 124, 0, 0.2) 100%);
    color: #ffa726;
    border: 1px solid #ffa726;
}
.badge-routine {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(56, 142, 60, 0.2) 100%);
    color: #81c784;
    border: 1px solid #4caf50;
}

/* Timeline */
.timeline-item {
    padding-left: 20px;
    border-left: 2px solid rgba(255, 255, 255, 0.1);
    position: relative;
    padding-bottom: 1.5rem;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -6px;
    top: 4px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #00c6ff;
}
.timeline-item.ed::before {
    background-color: #ffa500;
}

/* SDOH Pulsing Badges */
.sdoh-badge {
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sdoh-badge.positive {
    background: rgba(239, 83, 80, 0.1);
    border: 1px solid rgba(239, 83, 80, 0.3);
    color: #ef5350;
}
.sdoh-badge.negative {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    color: #888888;
}

</style>
""", unsafe_allow_html=True)

# Custom Header HTML
st.markdown("""
<div class="header-container">
    <div class="header-title">🏥 CareAgent Dashboard</div>
    <div class="header-subtitle">Clinical AI Agent System for 30-Day Hospital Readmission Risk Mitigation & SDOH Coordination</div>
</div>
""", unsafe_allow_html=True)

# Helper function to query backend
def fetch_patients(params=None):
    try:
        r = requests.get(f"{BACKEND_URL}/patients", params=params)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Failed to connect to CareAgent backend at {BACKEND_URL}. Ensure uvicorn server is running.")
    return None

def fetch_patient_detail(patient_id):
    try:
        r = requests.get(f"{BACKEND_URL}/patients/{patient_id}")
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Failed to fetch details for Patient {patient_id}.")
    return None

def trigger_discharge_event(encounter_id):
    try:
        r = requests.post(f"{BACKEND_URL}/events/discharge", json={"encounter_id": encounter_id})
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Failed to trigger discharge event for Encounter {encounter_id}.")
    return None

def send_chat_message(patient_id, msg):
    try:
        r = requests.post(f"{BACKEND_URL}/chat", json={"patient_id": patient_id, "message": msg})
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception as e:
        st.error("Failed to communicate with CareAgent AI.")
    return "Error generating response."

# --- Sidebar Filters & Patient Selector ---
st.sidebar.markdown("### 🔍 Patient Registry Filters")

risk_filter = st.sidebar.selectbox("Readmission Risk Band", ["All", "High", "Medium", "Low"])
care_filter = st.sidebar.selectbox("Care Management Level", ["All", "Intensive", "Enhanced", "Routine"])
sdoh_filter = st.sidebar.selectbox("SDOH Risk Level", ["All", "High", "Moderate", "Low"])
search_id = st.sidebar.text_input("Search Patient ID")

# Construct query params
params = {"limit": 5000} # Fetch all matching to show counts and selection
if risk_filter != "All":
    params["risk_band"] = risk_filter
if care_filter != "All":
    params["care_level"] = care_filter
if sdoh_filter != "All":
    params["sdoh_risk"] = sdoh_filter
if search_id.strip():
    params["search"] = search_id.strip()

# Fetch list of filtered patients
response_data = fetch_patients(params)

if response_data:
    patients = response_data.get("patients", [])
    total_found = response_data.get("total", 0)
    
    # Calculate overview stats from this batch
    high_risk_count = sum(1 for p in patients if p["readmit_risk_band"] == "High")
    intensive_care_count = sum(1 for p in patients if p["care_management_level"] == "Intensive")
    high_sdoh_count = sum(1 for p in patients if p["sdoh_risk_level"] == "High")
    
    # Render premium overview cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-lbl">Patients Monitored</p>
            <p class="metric-val">{total_found}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card critical">
            <p class="metric-lbl">High Readmission Risk</p>
            <p class="metric-val">{high_risk_count}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card warning">
            <p class="metric-lbl">Intensive Care Level</p>
            <p class="metric-val">{intensive_care_count}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card success">
            <p class="metric-lbl">High SDOH Barriers</p>
            <p class="metric-val">{high_sdoh_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.sidebar.markdown(f"**Found {total_found} matching patients.**")
    
    if patients:
        # Patient selector list
        patient_options = {f"Patient {p['patient_id']} (Risk: {p['readmit_risk_band']}, Care: {p['care_management_level']})": p['patient_id'] for p in patients}
        selected_option = st.sidebar.selectbox("Select Patient to Inspect", list(patient_options.keys()))
        selected_patient_id = patient_options[selected_option]
        
        # --- PATIENT DETAIL VIEW ---
        detail = fetch_patient_detail(selected_patient_id)
        if detail:
            profile = detail["profile"]
            latest_eval = detail["latest_evaluation"]
            encounters = profile.get("encounters", [])
            
            # Extract latest evaluation fields
            latest_run = latest_eval if "risk_results" in latest_eval else latest_eval.get("latest_evaluation", {})
            risk_results = latest_run.get("risk_results", {})
            recommendations = latest_run.get("recommendations", {})
            
            # Setup columns for Patient profile and risk panel
            left_col, right_col = st.columns([1, 2])
            
            with left_col:
                # Patient Demographics Card
                st.markdown(f"""
                <div class="glass-card">
                    <h3>👤 Patient Profile</h3>
                    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
                        <tr><td style="padding:6px 0; font-weight:500; opacity:0.8;">Patient ID</td><td style="text-align:right; font-weight:600;">{profile['patient_id']}</td></tr>
                        <tr><td style="padding:6px 0; font-weight:500; opacity:0.8;">Age / Sex</td><td style="text-align:right; font-weight:600;">{profile['age']} ({profile['sex']})</td></tr>
                        <tr><td style="padding:6px 0; font-weight:500; opacity:0.8;">Insurance</td><td style="text-align:right; font-weight:600;">{profile['insurance']}</td></tr>
                        <tr><td style="padding:6px 0; font-weight:500; opacity:0.8;">Preferred Language</td><td style="text-align:right; font-weight:600;">{profile['language']}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                # SDOH Status Card
                sdoh_score = profile.get("sdoh_score", 0)
                sdoh_lvl = profile.get("sdoh_risk_level", "Low")
                
                # Setup badge styling
                sdoh_class = "badge-high" if sdoh_lvl == "High" else ("badge-medium" if sdoh_lvl == "Moderate" else "badge-low")
                
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <h3 style="margin:0;">🧩 Social Barriers (SDOH)</h3>
                        <span class="badge {sdoh_class}">{sdoh_lvl} Risk</span>
                    </div>
                    <p style="font-size:0.9rem; opacity:0.8; margin-bottom:15px;">SDOH Score: <b>{sdoh_score} out of 4</b> barriers flagged.</p>
                """, unsafe_allow_html=True)
                
                # Render individual flags
                flags = [
                    ("Housing Instability", "housing_instability"),
                    ("Food Insecurity", "food_insecurity"),
                    ("Transportation Barriers", "transportation_barrier"),
                    ("Low Social Support", "low_social_support")
                ]
                
                for label, key in flags:
                    is_pos = profile.get(key) == 1
                    status_text = "🔴 Positive" if is_pos else "🟢 Negative"
                    status_class = "positive" if is_pos else "negative"
                    st.markdown(f"""
                    <div class="sdoh-badge {status_class}">
                        <span>{label}</span>
                        <span style="font-weight:600;">{status_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Simulation actions panel
                st.markdown("""
                <div class="glass-card">
                    <h3>⚙️ Simulators</h3>
                """, unsafe_allow_html=True)
                
                # Discharge a new encounter simulator
                st.markdown("<p style='font-size:0.9rem; opacity:0.8;'>Trigger CareAgent for a new discharge encounter:</p>", unsafe_allow_html=True)
                
                # Filter encounters that haven't been evaluated (just simulation helper)
                # To simulate, we show a button to run the agent on the latest encounter
                if encounters:
                    latest_enc = encounters[-1]
                    btn_lbl = f"Discharge Patient (Enc {latest_enc['encounter_id']})"
                    if st.button(btn_lbl, use_container_width=True):
                        with st.spinner("CareAgent processing discharge event..."):
                            res = trigger_discharge_event(latest_enc["encounter_id"])
                            if res and "error" not in res:
                                st.success(f"Successfully processed discharge! Risk: {res['risk_results']['readmit_risk_band']}.")
                                st.rerun()
                                
                st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                # Risk & Care Level Summary Card
                prob = risk_results.get("readmit_probability", 0.0)
                band = risk_results.get("readmit_risk_band", "Low")
                care_level = risk_results.get("care_management_level", "Routine")
                
                band_class = "badge-high" if band == "High" else ("badge-medium" if band == "Medium" else "badge-low")
                care_class = "badge-intensive" if care_level == "Intensive" else ("badge-enhanced" if care_level == "Enhanced" else "badge-routine")
                
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <h3 style="margin:0;">📊 CareAgent Readmission Analysis</h3>
                        <div>
                            <span class="badge {band_class}" style="margin-right:8px;">{band} Risk</span>
                            <span class="badge {care_class}">{care_level} Care</span>
                        </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:30px;">
                        <div style="text-align:center;">
                            <div style="font-size:3rem; font-weight:700; color:{'#ff4b4b' if band=='High' else '#ffa500' if band=='Medium' else '#00e676'};">{prob:.1%}</div>
                            <div style="font-size:0.8rem; text-transform:uppercase; opacity:0.7;">30-Day Probability</div>
                        </div>
                        <div style="border-left:1px solid rgba(255,255,255,0.1); padding-left:30px;">
                            <p style="margin:0 0 5px 0; font-weight:600; font-size:1.05rem;">Key Risk Drivers:</p>
                """, unsafe_allow_html=True)
                
                drivers = recommendations.get("risk_drivers", [])
                if drivers:
                    for d in drivers:
                        st.markdown(f"<li style='font-size:0.9rem; opacity:0.95;'>{d}</li>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size:0.9rem; opacity:0.7;'>No clinical risk drivers extracted yet. Click the discharge simulator to evaluate.</p>", unsafe_allow_html=True)
                    
                st.markdown(f"""
                        </div>
                    </div>
                    <p style="margin-top:20px; font-style:italic; font-size:0.95rem; line-height:1.5; opacity:0.85; border-top:1px solid rgba(255,255,255,0.05); padding-top:15px;">
                        <b>Clinical Rationale:</b> {recommendations.get('clinical_rationale', 'No rationale provided.')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations Checklists
                st.markdown("<h3 style='margin-top:1.5rem;'>📝 Actionable Recommendations</h3>", unsafe_allow_html=True)
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
                    st.markdown("<h4>📋 Clinical Interventions</h4>", unsafe_allow_html=True)
                    clinical_recs = recommendations.get("clinical_recommendations", [])
                    if clinical_recs:
                        for idx, r in enumerate(clinical_recs):
                            st.checkbox(r, key=f"clin_rec_{idx}")
                    else:
                        st.markdown("<p style='font-size:0.9rem; opacity:0.7;'>No clinical interventions generated.</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with rec_col2:
                    st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
                    st.markdown("<h4>🔗 SDOH Support Services</h4>", unsafe_allow_html=True)
                    sdoh_recs = recommendations.get("sdoh_interventions", [])
                    if sdoh_recs:
                        for idx, r in enumerate(sdoh_recs):
                            st.checkbox(r, key=f"sdoh_rec_{idx}")
                    else:
                        st.markdown("<p style='font-size:0.9rem; opacity:0.7;'>No SDOH support services required.</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Encounters History Table / Timeline
                st.markdown("<h3 style='margin-top:1.5rem;'>📅 Clinical Encounters History</h3>", unsafe_allow_html=True)
                if encounters:
                    enc_data = []
                    for e in encounters:
                        enc_data.append({
                            "Encounter ID": e["encounter_id"],
                            "Admit Day": e["admit_day"],
                            "Discharge Day": e["discharge_day"],
                            "LOS (Days)": e["length_of_stay"],
                            "Type": e["encounter_type"],
                            "Diagnosis": e["diagnosis_group"],
                            "Readmit 30d?": "🔴 Yes" if e["readmit_30"] == 1 else "🟢 No"
                        })
                    st.dataframe(pd.DataFrame(enc_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No prior encounters recorded for this patient.")

            # --- CHAT WITH CAREAGENT ---
            st.markdown("<h3 style='margin-top:2rem;'>💬 Chat with CareAgent Clinical Assistant</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.9rem; opacity:0.8; margin-bottom:1.5rem;'>Ask CareAgent about this patient's clinical barriers, social risks, or discharge readiness.</p>", unsafe_allow_html=True)
            
            # Initialize chat history in session state specific to the patient
            chat_state_key = f"chat_history_{selected_patient_id}"
            if chat_state_key not in st.session_state:
                st.session_state[chat_state_key] = []
                
            # Display past messages
            for message in st.session_state[chat_state_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            # User Input
            if prompt := st.chat_input("Ask CareAgent a question..."):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state[chat_state_key].append({"role": "user", "content": prompt})
                
                # Fetch agent response
                with st.chat_message("assistant"):
                    with st.spinner("CareAgent analyzing context..."):
                        response = send_chat_message(selected_patient_id, prompt)
                        st.markdown(response)
                st.session_state[chat_state_key].append({"role": "assistant", "content": response})
    else:
        st.info("No patients found matching the selected filters.")
else:
    st.warning("Please run the backend and data generator first to load patient data.")
