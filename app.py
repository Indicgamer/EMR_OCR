import streamlit as st
import os
import subprocess
import json
import pandas as pd
from PIL import Image
import uuid

# 1. MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="EMR AI Digitizer", layout="wide", page_icon="🏥")

# --- PATHS ---
OCR_SCRIPT = "medical_ocr.py"
LLM_SCRIPT = "emr_enginellm.py"

def run_pipeline(image_path):
    try:
        # Run OCR
        ocr_process = subprocess.run(
            ['python', OCR_SCRIPT, image_path],
            capture_output=True, text=True, check=True
        )
        ocr_text = ocr_process.stdout
        if not ocr_text.strip():
            return {"error": "OCR failed to extract text."}

        # Run LLM
        llm_process = subprocess.run(
            ['python', LLM_SCRIPT],
            input=ocr_text,
            capture_output=True, text=True, check=True,
            env=os.environ.copy()
        )
        return json.loads(llm_process.stdout)
    except Exception as e:
        return {"error": f"Pipeline Error: {str(e)}"}

# --- UI LAYOUT ---
st.title("🏥 Smart EMR OCR Digitizer")
st.markdown("#### MTech Project: Automated Clinical Pipeline")

# Robust Sidebar Implementation
with st.sidebar:
    st.header("📋 Instructions")
    st.write("1. Upload a clear image of a Lab Report.")
    st.write("2. Click 'Start Digitization'.")
    st.write("3. Verify the extracted FHIR data.")
    st.divider()
    st.info("System: PaddleOCR + Llama-3 (Groq)")

# Main Upload Area (More reliable than sidebar for tunnels)
uploaded_file = st.file_uploader("📤 Upload Lab Report Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preview and Process
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        img = Image.open(uploaded_file)
        st.image(img, caption="Source Document", use_container_width=True)
        temp_path = "temp_upload.png"
        img.save(temp_path)

    with col_info:
        if st.button("🚀 Start Digitization", use_container_width=True):
            with st.spinner("Processing through AI Pipeline..."):
                fhir_bundle = run_pipeline(temp_path)

            if "error" in fhir_bundle:
                st.error(fhir_bundle["error"])
            else:
                # --- DATA PARSING ---
                patient = {"name": "N/A", "age": "N/A", "sex": "N/A"}
                doctor = "N/A"
                observations = []

                for entry in fhir_bundle.get('entry', []):
                    res = entry.get('resource', {})
                    rtype = res.get('resourceType')

                    if rtype == 'Patient':
                        names = res.get('name', [{}])
                        patient['name'] = names[0].get('text') or names[0].get('family', 'N/A')
                        patient['sex'] = res.get('gender', 'N/A').capitalize()
                        patient['age'] = res.get('birthDate') or res.get('extension', [{}])[0].get('valueString', 'N/A')
                    elif rtype == 'Practitioner':
                        doctor = res.get('name', [{}])[0].get('text', 'N/A')
                    elif rtype == 'Observation':
                        test_name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
                        val_qty = res.get('valueQuantity', {})
                        val = val_qty.get('value', res.get('valueString', 'N/A'))
                        unit = val_qty.get('unit', '')
                        observations.append({"Parameter": test_name, "Result": f"{val} {unit}", "LOINC": res.get('code', {}).get('coding', [{}])[0].get('code', 'N/A')})

                # --- DASHBOARD ---
                st.success("Analysis Complete!")
                st.subheader("📋 Patient Metadata")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Patient", patient['name'])
                m2.metric("Age", patient['age'])
                m3.metric("Gender", patient['sex'])
                m4.metric("Doctor", doctor)

                st.divider()
                st.subheader("🧪 Laboratory Observations")
                if observations:
                    st.table(pd.DataFrame(observations))
                else:
                    st.warning("No lab results detected.")

                st.download_button("📩 Download FHIR JSON", data=json.dumps(fhir_bundle, indent=2), file_name="EMR_Record.json")

else:
    st.warning("Waiting for file upload...")