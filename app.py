import streamlit as st
import os
import subprocess
import json
import pandas as pd
from PIL import Image
from datetime import datetime
import uuid

# --- CONFIGURATION ---
st.set_page_config(page_title="EMR AI Digitizer", layout="wide", page_icon="🏥")

# Path to your existing scripts (assumes they are in the same directory as app.py)
OCR_SCRIPT = "medical_ocr.py"
LLM_SCRIPT = "emr_enginellm.py"

def run_pipeline(image_path):
    """
    Calls the OCR and LLM scripts internally using subprocess.
    Equivalent to: python medical_ocr.py img | python emr_enginellm.py
    """
    try:
        # 1. Run the Layout-Aware OCR Script
        # We capture the stdout which contains the cleaned OCR text
        ocr_process = subprocess.run(
            ['python', OCR_SCRIPT, image_path],
            capture_output=True, 
            text=True, 
            check=True
        )
        ocr_text = ocr_process.stdout

        if not ocr_text.strip():
            return {"error": "OCR failed to extract text from the image."}

        # 2. Run the LLM Parser Script
        # We pass the ocr_text into the stdin of the LLM script
        llm_process = subprocess.run(
            ['python', LLM_SCRIPT],
            input=ocr_text,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy() # Pass the GROQ_API_KEY from environment
        )
        
        return json.loads(llm_process.stdout)

    except subprocess.CalledProcessError as e:
        return {"error": f"Pipeline Error: {e.stderr}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}

# --- STREAMLIT UI ---
st.title("🏥 Smart EMR OCR Digitizer")
st.markdown("#### MTech Project: Automated Clinical Information Extraction Pipeline")
st.info("This system uses PaddleOCR for layout analysis and Llama-3 (Groq) for Medical Entity Recognition.")

# Sidebar for Image Upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a Lab Report image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_image_path = "temp_upload.png"
    img = Image.open(uploaded_file)
    img.save(temp_image_path)
    
    # Show the image in sidebar
    st.sidebar.image(img, caption="Source Image", use_container_width=True)

    if st.button("🚀 Start Digitization"):
        with st.spinner("Processing... Calling OCR and LLM Engines..."):
            # Execute the internal scripts
            fhir_bundle = run_pipeline(temp_image_path)

        if "error" in fhir_bundle:
            st.error(fhir_bundle["error"])
        else:
            # --- EXTRACTION LOGIC ---
            patient = {"name": "N/A", "age": "N/A", "sex": "N/A"}
            doctor = "N/A"
            observations = []

            for entry in fhir_bundle.get('entry', []):
                res = entry.get('resource', {})
                rtype = res.get('resourceType')

                # Extract Patient Data
                if rtype == 'Patient':
                    names = res.get('name', [{}])[0]
                    patient['name'] = names.get('text') or names.get('family', 'N/A')
                    patient['sex'] = res.get('gender', 'N/A').capitalize()
                    patient['age'] = res.get('birthDate') or res.get('extension', [{}])[0].get('valueString', 'N/A')
                
                # Extract Doctor Data
                elif rtype == 'Practitioner':
                    doctor = res.get('name', [{}])[0].get('text', 'N/A')
                
                # Extract Lab Results
                elif rtype == 'Observation':
                    test_name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown Test')
                    loinc = res.get('code', {}).get('coding', [{}])[0].get('code', 'N/A')
                    
                    val_qty = res.get('valueQuantity', {})
                    val_str = res.get('valueString', '')
                    value = val_qty.get('value', val_str)
                    unit = val_qty.get('unit', '')
                    
                    observations.append({
                        "Test Description": test_name,
                        "Result": f"{value} {unit}",
                        "LOINC Code": loinc
                    })

            # --- DISPLAY DASHBOARD ---
            st.markdown("### 📋 Patient & Clinical Metadata")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Patient Name", patient['name'])
            m2.metric("Age/DOB", patient['age'])
            m3.metric("Gender", patient['sex'])
            m4.metric("Ordering Doctor", doctor)

            st.markdown("---")
            st.subheader("🧪 Extracted Laboratory Observations")
            
            if observations:
                df = pd.DataFrame(observations)
                st.table(df)
            else:
                st.warning("No lab results could be extracted.")

            # Download Options
            st.markdown("---")
            st.download_button(
                label="📩 Download FHIR JSON Bundle",
                data=json.dumps(fhir_bundle, indent=2),
                file_name=f"EMR_{uuid.uuid4().hex[:8]}.json",
                mime="application/json"
            )
            st.success("Digitization Complete. Data follows HL7 FHIR Standards.")

else:
    st.write("Please upload a lab report image from the sidebar to begin.")

# Cleanup temp file on close
if os.path.exists("temp_upload.png"):
    pass # Managed by streamlit sessions usually