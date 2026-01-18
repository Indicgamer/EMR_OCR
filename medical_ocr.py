import streamlit as st
import os
import json
import pandas as pd
from PIL import Image

# Set Environment Variables BEFORE importing PaddleOCR
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Try to import your logic directly
try:
    from medical_ocr import MedicalLayoutOCR
    from emr_enginellm import GroqEMREngine
except ImportError:
    st.error("Missing medical_ocr.py or emr_enginellm.py in the repository.")

# 1. Page Config
st.set_page_config(page_title="EMR AI Digitizer", layout="wide", page_icon="🏥")

# 2. Cache the Engines so they don't reload on every click
@st.cache_resource
def load_engines():
    ocr = MedicalLayoutOCR()
    llm = GroqEMREngine()
    return ocr, llm

# --- UI LAYOUT ---
st.title("🏥 Smart EMR OCR Digitizer")
st.markdown("#### MTech Project: Automated Clinical Pipeline")

# Main Upload Area
uploaded_file = st.file_uploader("📤 Upload Lab Report Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_img, col_info = st.columns([1, 1])
    
    # Save and Preview
    temp_path = "temp_upload.png"
    img = Image.open(uploaded_file)
    img.save(temp_path)
    
    with col_img:
        st.image(img, caption="Source Document", use_container_width=True)

    with col_info:
        if st.button("🚀 Start Digitization", use_container_width=True):
            try:
                with st.spinner("Initializing AI Engines (First run may take a moment)..."):
                    ocr_engine, llm_engine = load_engines()

                # Step 1: OCR
                with st.spinner("Extracting Layout & Text..."):
                    raw_text = ocr_engine.get_layout_rows(temp_path)
                
                if not raw_text.strip():
                    st.error("OCR returned empty text. Please ensure the image is clear.")
                else:
                    # Step 2: LLM Parsing
                    with st.spinner("Processing Clinical Entities with Llama-3..."):
                        fhir_bundle = llm_engine.process_ocr(raw_text)

                    if "error" in fhir_bundle:
                        st.error(f"LLM Error: {fhir_bundle['error']}")
                    else:
                        # --- DISPLAY RESULTS ---
                        st.success("Analysis Complete!")
                        
                        # Extract Patient Info
                        patient_name = "N/A"
                        doctor_name = "N/A"
                        observations = []

                        for entry in fhir_bundle.get('entry', []):
                            res = entry.get('resource', {})
                            rtype = res.get('resourceType')
                            if rtype == 'Patient':
                                patient_name = res.get('name', [{}])[0].get('text', 'N/A')
                            elif rtype == 'Practitioner':
                                doctor_name = res.get('name', [{}])[0].get('text', 'N/A')
                            elif rtype == 'Observation':
                                name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Test')
                                val = res.get('valueQuantity', {}).get('value', 'N/A')
                                unit = res.get('valueQuantity', {}).get('unit', '')
                                observations.append({"Parameter": name, "Result": f"{val} {unit}"})

                        # Show Info
                        st.write(f"**Patient:** {patient_name} | **Doctor:** {doctor_name}")
                        
                        if observations:
                            st.table(pd.DataFrame(observations))
                        
                        st.download_button(
                            "📩 Download FHIR JSON", 
                            data=json.dumps(fhir_bundle, indent=2), 
                            file_name="clinical_record.json"
                        )
            except Exception as e:
                st.error(f"System Error: {str(e)}")
else:
    st.info("Please upload a report to begin.")