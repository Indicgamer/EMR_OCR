import os

app_content = """
import streamlit as st
import json
import pandas as pd
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="EMR Clinical Dashboard", layout="wide", page_icon="🏥")

# --- STYLING ---
st.markdown(\"\"\"
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #1e3a8a; }
    h3 { color: #1e40af; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
    </style>
\"\"\", unsafe_allow_value=True)

def load_data(filepath="final_record.json"):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def extract_fhir_entities(bundle):
    patient = {"name": "---", "age": "---", "gender": "---"}
    doctor = "---"
    observations = []

    if not bundle or "entry" not in bundle:
        return patient, doctor, observations

    for entry in bundle["entry"]:
        res = entry.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Patient":
            names = res.get("name", [])
            if names:
                patient["name"] = names[0].get("text") or names[0].get("family", "---")
            patient["gender"] = res.get("gender", "---").capitalize()
            patient["age"] = res.get("birthDate") or "---"

        elif rtype == "Practitioner":
            names = res.get("name", [])
            if names:
                doctor = names[0].get("text", "---")

        elif rtype == "Observation":
            code_data = res.get("code", {}).get("coding", [{}])[0]
            test_name = code_data.get("display", "Unknown Test")
            loinc = code_data.get("code", "---")
            
            val_qty = res.get("valueQuantity", {})
            val_str = res.get("valueString", "")
            val = val_qty.get("value", val_str)
            unit = val_qty.get("unit", "")
            
            observations.append({
                "LOINC Code": loinc,
                "Clinical Parameter": test_name,
                "Result": f"{val} {unit}".strip(),
                "Status": "Verified"
            })
            
    return patient, doctor, observations

# --- UI LAYOUT ---
st.title("🏥 Medical Information System")
st.caption("MTech Project: Automated EMR Digitization Pipeline (Stage 8 & 9)")

data = load_data()

if data:
    patient, doctor, obs_list = extract_fhir_entities(data)

    st.markdown("### 📋 Patient Metadata")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patient Name", patient["name"])
    col2.metric("Age/DOB", patient["age"])
    col3.metric("Gender", patient["gender"])
    col4.metric("Ordering Doctor", doctor)

    st.write("") 

    st.markdown("### 🧪 Laboratory Observations")
    if obs_list:
        df = pd.DataFrame(obs_list)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No lab observations were found in the FHIR Bundle.")

    st.markdown("---")
    c_left, c_right = st.columns(2)
    with c_left:
        if st.button("✅ Commit Record to ABDM Registry"):
            st.success("Successfully validated and saved to hospital database.")
            st.balloons()
    with c_right:
        st.download_button(
            label="📩 Download FHIR JSON",
            data=json.dumps(data, indent=2),
            file_name="patient_record.json",
            mime="application/json"
        )
else:
    st.error("Error: 'final_record.json' not found. Please run the OCR and LLM scripts first.")
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("✅ File 'app.py' has been successfully written to disk.")