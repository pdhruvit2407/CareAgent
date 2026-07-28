"""
===============================================================================
CAREAGENT: STREAMLIT CLINICAL DASHBOARD (frontend.py)
===============================================================================
PURPOSE:
This module provides a state-of-the-art Streamlit web interface for care coordinators,
clinicians, and health system managers. It renders dynamic risk gauges, PRAPARE SDOH
hover tooltips, interactive post-discharge checklists, longitudinal history timelines,
EHR discharge simulators, and an inline Gemini clinical assistant chatbot.

KEY USER INTERFACE FEATURES:
1. Patient Registry Filters : Filter patients by Risk Band, Care Tier, SDOH Level, ID, or Starred Status.
2. Reset Filters Callback   : Native Streamlit callback button to clear all filters without lifecycle errors.
3. Patient Profile & SDOH   : Displays demographics and PRAPARE screening question hover tooltips.
4. Risk Gauge & Drivers     : Visualizes 30-day readmission risk probability and AI clinical risk drivers.
5. Interactive Checklists  : Persists completed clinical actions & SDOH interventions to backend storage.
6. Chatbot Assistant Form   : Inline Streamlit form with clear_on_submit=True to query Gemini on patient context.
===============================================================================
"""

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

# Initialize session state for filters if not present
if "filter_risk" not in st.session_state:
    st.session_state.filter_risk = "All"
if "filter_care" not in st.session_state:
    st.session_state.filter_care = "All"
if "filter_sdoh" not in st.session_state:
    st.session_state.filter_sdoh = "All"
if "filter_diag" not in st.session_state:
    st.session_state.filter_diag = "All"
if "filter_age_category" not in st.session_state:
    st.session_state.filter_age_category = "All"
if "filter_search" not in st.session_state:
    st.session_state.filter_search = ""
if "filter_monitored" not in st.session_state:
    st.session_state.filter_monitored = False

risk_filter = st.sidebar.selectbox("Readmission Risk Band", ["All", "High", "Medium", "Low"], key="filter_risk")
care_filter = st.sidebar.selectbox("Care Management Level", ["All", "Intensive", "Enhanced", "Routine"], key="filter_care")
sdoh_filter = st.sidebar.selectbox("SDOH Risk Level", ["All", "High", "Moderate", "Low"], key="filter_sdoh")
diag_filter = st.sidebar.selectbox("Diagnosis Cohort", ["All", "CHF", "COPD", "Diabetes", "Asthma", "Hypertension"], key="filter_diag")
age_cat_filter = st.sidebar.selectbox("Age Category", ["All", "Pediatric (<= 18)", "Adult (> 18)"], key="filter_age_category")
search_id = st.sidebar.text_input("Search Patient ID", key="filter_search")
monitored_filter = st.sidebar.checkbox("⭐ Monitored Patients Only", key="filter_monitored")

def reset_filters():
    st.session_state.filter_risk = "All"
    st.session_state.filter_care = "All"
    st.session_state.filter_sdoh = "All"
    st.session_state.filter_diag = "All"
    st.session_state.filter_age_category = "All"
    st.session_state.filter_search = ""
    st.session_state.filter_monitored = False

st.sidebar.button("🏠 Reset Filters", on_click=reset_filters, use_container_width=True)

# Construct query params
params = {"limit": 5000} # Fetch all matching to show counts and selection
if risk_filter != "All":
    params["risk_band"] = risk_filter
if care_filter != "All":
    params["care_level"] = care_filter
if sdoh_filter != "All":
    params["sdoh_risk"] = sdoh_filter
if diag_filter != "All":
    params["diagnosis"] = diag_filter
if age_cat_filter == "Pediatric (<= 18)":
    params["age_category"] = "pediatric"
elif age_cat_filter == "Adult (> 18)":
    params["age_category"] = "adult"
if search_id.strip():
    params["search"] = search_id.strip()
if monitored_filter:
    params["monitored"] = True

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
            is_monitored = detail.get("monitored", False)
            
            # Monitoring Action Header
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                st.markdown(f"<h2 style='margin-bottom:0;'>👤 Patient {selected_patient_id} Detail View</h2>", unsafe_allow_html=True)
            with col_btn:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                btn_lbl = "⭐ Monitored" if is_monitored else "☆ Monitor Patient"
                if st.button(btn_lbl, key=f"mon_toggle_{selected_patient_id}", use_container_width=True):
                    try:
                        requests.post(f"{BACKEND_URL}/patients/{selected_patient_id}/monitor", json={"monitored": not is_monitored})
                        st.rerun()
                    except Exception as e:
                        st.error("Failed to update monitoring status.")
            
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
                     <p style="font-size:0.9rem; opacity:0.8; margin-bottom:15px;">SDOH Score: <b>{sdoh_score} out of 6</b> barriers flagged.</p>
                """, unsafe_allow_html=True)
                
                # Render individual flags with screening question hover tooltips (all 6 categories)
                flags = [
                    (
                        "Food Insecurity", 
                        "food_insecurity", 
                        "In the past 12 months, did you worry whether your food would run out before you got money to buy more?"
                    ),
                    (
                        "Income / Financial Strain", 
                        "income_barrier", 
                        "Do you experience difficulty paying for basic necessities like medicine, housing, or food at the end of the month?"
                    ),
                    (
                        "Housing Instability", 
                        "housing_instability", 
                        "In the past 12 months, have you worried about losing your housing, or lived in a shelter or temporary housing?"
                    ),
                    (
                        "Education & Literacy", 
                        "education_literacy_barrier", 
                        "Do you ever need help reading or understanding medical instructions or pamphlets given by doctors?"
                    ),
                    (
                        "Social Isolation", 
                        "low_social_support", 
                        "How often do you feel you have someone you can count on to help you if you get sick or need assistance?"
                    ),
                    (
                        "Transportation Barriers", 
                        "transportation_barrier", 
                        "In the past 12 months, has a lack of transportation kept you from medical appointments, meetings, or prescriptions?"
                    )
                ]
                
                for label, key, question in flags:
                    is_pos = profile.get(key) == 1
                    status_text = "🔴 Positive" if is_pos else "🟢 Negative"
                    status_class = "positive" if is_pos else "negative"
                    st.markdown(f"""
                    <div class="sdoh-badge {status_class}" title="{question}" style="cursor: help;">
                        <span>{label} ℹ️</span>
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
                # Horizons values
                prob_30 = risk_results.get("readmit_probability", 0.0)
                band_30 = risk_results.get("readmit_risk_band", "Low")
                prob_60 = risk_results.get("readmit_probability_60", prob_30 + 0.12)
                band_60 = risk_results.get("readmit_risk_band_60", "Medium")
                prob_90 = risk_results.get("readmit_probability_90", prob_60 + 0.08)
                band_90 = risk_results.get("readmit_risk_band_90", "High")
                
                care_level = risk_results.get("care_management_level", "Routine")
                care_class = "badge-intensive" if care_level == "Intensive" else ("badge-enhanced" if care_level == "Enhanced" else "badge-routine")
                
                # Colors based on band
                def get_color(b):
                    return '#ff4b4b' if b == 'High' else '#ffa500' if b == 'Medium' else '#00e676'
                
                # Construct Key Risk Drivers HTML block
                drivers = recommendations.get("risk_drivers", [])
                drivers_html = ""
                if drivers:
                    for d in drivers:
                        drivers_html += f"<li style='font-size:0.9rem; opacity:0.95; margin-bottom:5px;'>{d}</li>"
                else:
                    drivers_html = "<p style='font-size:0.9rem; opacity:0.7; margin:0;'>No clinical risk drivers extracted yet. Click the discharge simulator to evaluate.</p>"
                
                # Construct Local Feature Contributions HTML block
                FEAT_MAP = {
                    "age": "Age",
                    "sex": "Sex",
                    "insurance": "Insurance Type",
                    "language": "Preferred Language",
                    "food_insecurity": "Food Insecurity",
                    "income_barrier": "Income Barrier",
                    "housing_instability": "Housing Instability",
                    "education_literacy_barrier": "Education / Literacy Barrier",
                    "low_social_support": "Low Social Support",
                    "transportation_barrier": "Transportation Barrier",
                    "sdoh_score": "Total SDOH Count",
                    "sdoh_risk_level": "SDOH Risk Level",
                    "length_of_stay": "Length of Stay",
                    "encounter_type": "Encounter Type",
                    "diagnosis_group": "Diagnosis Group",
                    "prior_encounters": "Prior Encounters (12m)",
                    "prior_ed": "Prior ED Visits (12m)",
                    "prior_inpatient": "Prior Inpatient Stays (12m)"
                }
                
                contrib_html = ""
                local_contribs = risk_results.get("local_contributions", {}).get("30_day", {})
                if local_contribs:
                    sorted_contribs = sorted([(k, v) for k, v in local_contribs.items() if abs(v) >= 0.005], key=lambda x: abs(x[1]), reverse=True)[:5]
                    if sorted_contribs:
                        max_abs = max(abs(v) for _, v in sorted_contribs)
                        if max_abs == 0: max_abs = 1.0
                        
                        contrib_rows = ""
                        for feat, val in sorted_contribs:
                            label = FEAT_MAP.get(feat, feat)
                            pct_val = val * 100
                            pct_text = f"+{pct_val:.1f}%" if val > 0 else f"{pct_val:.1f}%"
                            color = "#ff4b4b" if val > 0 else "#00e676"
                            bar_color = "linear-gradient(90deg, #ff4b4b, #ff7676)" if val > 0 else "linear-gradient(90deg, #00e676, #66ffa6)"
                            bar_width = (abs(val) / max_abs) * 100
                            
                            contrib_rows += f"""
                            <div style="margin-bottom:10px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px;">
                                    <span>{label}</span>
                                    <span style="font-weight:600; color:{color};">{pct_text}</span>
                                </div>
                                <div style="background:rgba(255,255,255,0.04); border-radius:4px; height:8px; width:100%; border: 1px solid rgba(255,255,255,0.02);">
                                    <div style="background:{bar_color}; width:{bar_width:.1f}%; height:100%; border-radius:4px;"></div>
                                </div>
                            </div>
                            """
                            
                        contrib_html = f"""
                        <div style="margin-top:20px; border-top:1px solid rgba(255,255,255,0.05); padding-top:15px;">
                            <h4 style="margin:0 0 10px 0;">👤 Machine Learning Feature Contributions</h4>
                            <p style="font-size:0.78rem; opacity:0.7; margin-bottom:12px; line-height:1.4;">
                                Mathematical contributions of patient-specific attributes to the 30-Day Risk prediction. Red indicates factors increasing risk; Green indicates protective factors.
                            </p>
                            {contrib_rows}
                        </div>
                        """
                
                # Generate demographic/historical baseline comparison context
                age = profile.get("age", 50)
                ins = profile.get("insurance", "Medicare")
                prior_encs = len(profile.get("encounters", [])) - 1
                sdoh_score = profile.get("sdoh_score", 0)
                
                # Check actual risk bands
                is_30_high = band_30 == "High"
                is_60_high = band_60 == "High"
                is_90_high = band_90 == "High"
                
                comp_reasons = []
                
                # 30-Day Risk Context
                if is_30_high:
                    if prior_encs >= 3:
                        comp_reasons.append(f"high utilization history ({prior_encs} prior encounters) directly contributes to their high Days 1–30 risk band ({prob_30:.1%})")
                    else:
                        comp_reasons.append(f"clinical complexity from the current acute encounter escalates their short-term Days 1–30 risk to {prob_30:.1%}")
                else:
                    if prior_encs == 0:
                        comp_reasons.append(f"lack of prior hospitalizations helps maintain a low baseline risk in the initial 30 days ({prob_30:.1%})")
                    else:
                        comp_reasons.append(f"stabilized transition care factors help moderate their initial 30-day readmission risk ({prob_30:.1%})")
                
                # 60-Day Risk Context
                if is_60_high:
                    if ins in ["Medicaid", "Uninsured"]:
                        comp_reasons.append(f"access-to-care barriers associated with their enrollment status ({ins}) elevate their mid-term Days 31–60 risk to {prob_60:.1%}")
                    else:
                        comp_reasons.append(f"clinical outpatient transition challenges keep their mid-term risk elevated at {prob_60:.1%}")
                else:
                    comp_reasons.append(f"outpatient adherence checks help stabilize their Days 31–60 risk at a moderate or low level ({prob_60:.1%})")
                
                # 90-Day Risk Context
                if is_90_high:
                    if sdoh_score >= 3:
                        comp_reasons.append(f"severe SDOH burden ({sdoh_score}/6 flags positive) creates long-term compliance challenges, causing their Days 61–90 risk to escalate to {prob_90:.1%}")
                    elif age >= 65:
                        comp_reasons.append(f"advanced age ({age}) and related chronic needs elevate their long-term 90-day risk profile to {prob_90:.1%}")
                    else:
                        comp_reasons.append(f"long-term transition factors result in an elevated 90-day risk ({prob_90:.1%})")
                else:
                    reasons_low_90 = []
                    if age < 65:
                        reasons_low_90.append(f"younger age ({age}) which generally provides better physiological reserve")
                    if sdoh_score < 3:
                        reasons_low_90.append(f"manageable social needs score ({sdoh_score}/6)")
                    else:
                        reasons_low_90.append("protective clinical factors which override their social needs")
                        
                    desc_low_90 = ", combined with ".join(reasons_low_90)
                    comp_reasons.append(f"long-term Days 61–90 risk remains low ({prob_90:.1%}) due to {desc_low_90}")
                
                comp_text = f"Patient is a {age}-year-old with {ins} coverage. "
                if comp_reasons:
                    comp_text += "Analysis shows that their " + "; and their ".join(comp_reasons) + "."
                else:
                    comp_text += "Their risk scores align closely with standard demographic baseline averages for this cohort."
                
                # Compile complete HTML container for the timeline card
                timeline_card_html = f"""<div class="glass-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
<h3 style="margin:0;">📊 CareAgent Readmission Risk Timeline</h3>
<span class="badge {care_class}">{care_level} Care Tier</span>
</div>

<div>
<!-- 30-Day Risk -->
<div style="margin-bottom:12px; cursor:help;" title="30-Day Risk (Days 1–30): Reflects immediate post-discharge instability, medication errors, and transition care gaps.">
<div style="display:flex; justify-content:space-between; font-size:0.95rem; font-weight:600; margin-bottom:4px;">
<span>30-Day Risk (Primary Deciding Factor) ℹ️</span>
<span style="color:{get_color(band_30)};">{prob_30:.1%} ({band_30})</span>
</div>
<div style="background:rgba(255,255,255,0.08); border-radius:8px; height:12px; overflow:hidden; border: 1px solid rgba(255,255,255,0.05);">
<div style="background:{get_color(band_30)}; width:{prob_30:.1%}; height:100%;"></div>
</div>
</div>

<!-- 60-Day Risk -->
<div style="margin-bottom:12px; cursor:help;" title="60-Day Risk (Days 31–60): Reflects outpatient follow-up adherence, compliance with chronic care pathway, and lifestyle adherence.">
<div style="display:flex; justify-content:space-between; font-size:0.9rem; font-weight:500; opacity:0.9; margin-bottom:4px;">
<span>60-Day Risk ℹ️</span>
<span style="color:{get_color(band_60)};">{prob_60:.1%} ({band_60})</span>
</div>
<div style="background:rgba(255,255,255,0.08); border-radius:8px; height:8px; overflow:hidden; border: 1px solid rgba(255,255,255,0.05);">
<div style="background:{get_color(band_60)}; width:{prob_60:.1%}; height:100%;"></div>
</div>
</div>

<!-- 90-Day Risk -->
<div style="margin-bottom:5px; cursor:help;" title="90-Day Risk (Days 61–90): Reflects long-term social barrier impact (SDOH), financial barriers, and primary care access stability.">
<div style="display:flex; justify-content:space-between; font-size:0.9rem; font-weight:500; opacity:0.9; margin-bottom:4px;">
<span>90-Day Risk ℹ️</span>
<span style="color:{get_color(band_90)};">{prob_90:.1%} ({band_90})</span>
</div>
<div style="background:rgba(255,255,255,0.08); border-radius:8px; height:8px; overflow:hidden; border: 1px solid rgba(255,255,255,0.05);">
<div style="background:{get_color(band_90)}; width:{prob_90:.1%}; height:100%;"></div>
</div>
</div>
</div>

<div style="margin-top:15px; padding:12px; background:rgba(255,255,255,0.02); border-radius:6px; border:1px solid rgba(255,255,255,0.05); font-size:0.88rem; line-height:1.5;">
<b style="color:#2196f3;">🔬 Demographic & Clinical Baseline Context:</b><br/>
{comp_text}
</div>

<div style="margin-top:20px; border-top:1px solid rgba(255,255,255,0.05); padding-top:15px;">
<h4 style="margin:0 0 10px 0;">🔑 Key Risk Drivers</h4>
<ul style="margin:0; padding-left:20px;">
{drivers_html}
</ul>
</div>

<p style="margin-top:20px; font-style:italic; font-size:0.95rem; line-height:1.5; opacity:0.85; border-top:1px solid rgba(255,255,255,0.05); padding-top:15px;">
<b>Clinical Rationale:</b> {recommendations.get('clinical_rationale', 'No rationale provided.')}
</p>
{contrib_html}
</div>"""
                st.markdown(timeline_card_html, unsafe_allow_html=True)
                
                # Recommendations Checklists
                st.markdown("<h3 style='margin-top:1.5rem;'>📝 Actionable Recommendations</h3>", unsafe_allow_html=True)
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
                    st.markdown("<h4>📋 Clinical Interventions</h4>", unsafe_allow_html=True)
                    clinical_recs = recommendations.get("clinical_recommendations", [])
                    if clinical_recs:
                        for idx, r in enumerate(clinical_recs):
                            if isinstance(r, dict):
                                text = r.get("text", "")
                                completed = r.get("completed", False)
                            else:
                                text = r
                                completed = False
                                
                            key = f"clin_{selected_patient_id}_{idx}"
                            checked = st.checkbox(text, value=completed, key=key)
                            
                            if checked != completed:
                                try:
                                    requests.post(
                                        f"{BACKEND_URL}/patients/{selected_patient_id}/checklist",
                                        json={"category": "clinical", "index": idx, "completed": checked}
                                    )
                                    st.rerun()
                                except Exception:
                                    pass
                    else:
                        st.markdown("<p style='font-size:0.9rem; opacity:0.7;'>No clinical interventions generated.</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with rec_col2:
                    st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
                    st.markdown("<h4>🔗 SDOH Support Services</h4>", unsafe_allow_html=True)
                    sdoh_recs = recommendations.get("sdoh_interventions", [])
                    if sdoh_recs:
                        for idx, r in enumerate(sdoh_recs):
                            if isinstance(r, dict):
                                text = r.get("text", "")
                                completed = r.get("completed", False)
                            else:
                                text = r
                                completed = False
                                
                            key = f"sdoh_{selected_patient_id}_{idx}"
                            checked = st.checkbox(text, value=completed, key=key)
                            
                            if checked != completed:
                                try:
                                    requests.post(
                                        f"{BACKEND_URL}/patients/{selected_patient_id}/checklist",
                                        json={"category": "sdoh", "index": idx, "completed": checked}
                                    )
                                    st.rerun()
                                except Exception:
                                    pass
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
                    
            # User Input (using stable inline inputs and forms to avoid viewport-pinning and lifecycle issues)
            input_key = f"chat_input_val_{selected_patient_id}"
            with st.form(key=f"chat_form_{selected_patient_id}", clear_on_submit=True):
                chat_col, btn_col = st.columns([5, 1])
                with chat_col:
                    user_msg = st.text_input("Ask CareAgent a question...", placeholder="Type your question here...", key=input_key, label_visibility="collapsed")
                with btn_col:
                    submit_chat = st.form_submit_button("Send", use_container_width=True)
                    
            if submit_chat and user_msg.strip():
                prompt = user_msg.strip()
                
                # Update session state with user message
                st.session_state[chat_state_key].append({"role": "user", "content": prompt})
                
                # Fetch agent response
                with st.spinner("CareAgent analyzing context..."):
                    response = send_chat_message(selected_patient_id, prompt)
                st.session_state[chat_state_key].append({"role": "assistant", "content": response})
                st.rerun()
    else:
        st.info("No patients found matching the selected filters.")
else:
    st.warning("Please run the backend and data generator first to load patient data.")
