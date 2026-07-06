import os
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_generator import generate_synthetic_data
from src.model import train_baseline_model
from src.agent import CareAgentOrchestrator

def run_verification():
    print("==================================================")
    print("          CAREGENT SYSTEM VERIFICATION            ")
    print("==================================================")
    
    # 1. Generate Synthetic Data
    print("\n[STEP 1] Generating synthetic healthcare data...")
    generate_synthetic_data(data_dir="data", num_patients=500)  # Use 500 patients for fast verification
    
    # Verify CSV files exist
    files = ["careagent_patients_5000.csv", "careagent_encounters_5000.csv", "careagent_sdoh_5000.csv"]
    for f in files:
        path = os.path.join("data", f)
        if os.path.exists(path):
            print(f"  - Verified: {f} exists ({os.path.getsize(path)} bytes)")
        else:
            print(f"  - ERROR: {f} is missing!")
            return False
            
    # 2. Train baseline model
    print("\n[STEP 2] Training baseline readmission model...")
    train_baseline_model(data_dir="data", model_dir="src")
    
    # Verify model artifact exists
    model_path = "src/model_artifacts.pkl"
    if os.path.exists(model_path):
        print(f"  - Verified: model_artifacts.pkl exists ({os.path.getsize(model_path)} bytes)")
    else:
        print("  - ERROR: model_artifacts.pkl is missing!")
        return False
        
    # 3. Instantiate Agent Orchestrator and run a test discharge event
    print("\n[STEP 3] Running CareAgent Orchestrator test discharge event...")
    orchestrator = CareAgentOrchestrator(data_dir="data", model_path="src/model_artifacts.pkl")
    
    # Fetch a sample encounter
    sample_enc_id = 100001
    print(f"  - Processing discharge event for Encounter ID: {sample_enc_id}...")
    result = orchestrator.process_discharge_event(sample_enc_id)
    
    if "error" in result:
        print(f"  - ERROR processing discharge: {result['error']}")
        return False
        
    print("\n[VERIFICATION SUCCESS] CareAgent evaluation completed successfully!")
    print(f"  - Patient ID: {result['patient_id']}")
    print(f"  - Age / Sex: {result['profile']['age']} / {result['profile']['sex']}")
    print(f"  - Predicted Risk Probability: {result['risk_results']['readmit_probability']:.2%}")
    print(f"  - Predicted Risk Band: {result['risk_results']['readmit_risk_band']}")
    print(f"  - Care Management Level: {result['risk_results']['care_management_level']}")
    print("\nRecommendations Preview:")
    print("  - Clinical:")
    for r in result["recommendations"]["clinical_recommendations"][:2]:
        print(f"    * {r}")
    print("  - SDOH:")
    for r in result["recommendations"]["sdoh_interventions"][:2]:
        print(f"    * {r}")
    print(f"  - Rationale: {result['recommendations']['clinical_rationale']}")
    
    print("\nMemory Check:")
    history = orchestrator.memory.get_history(result["patient_id"])
    print(f"  - Patient {result['patient_id']} has {len(history)} run(s) saved in persistent memory.")
    
    print("\n==================================================")
    print("       SYSTEM VERIFICATION COMPLETE (PASS)        ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
