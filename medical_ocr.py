import streamlit as st
import os
import subprocess
import json
import pandas as pd
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="EMR AI Digitizer", layout="wide", page_icon="🏥")

# 2. Pipeline Function
def run_pipeline(image_path):
    try:
        # OCR
        ocr_process = subprocess.run(['python', 'medical_ocr.py', image_path], capture_output=True, text=True, check=True)
        ocr_text = ocr_process.stdout
        
        # LLM
        llm_process = subprocess.run(['python', 'emr_enginellm.py'], input=ocr_text, capture_output=True, text=True, check=True, env=os.environ.copy())
        return json.loads(llm_process.stdout)
    except Exception as e:
        return {"error": str(e)}

# 3. Main UI
st.title("🏥 Medical Record Digitizer")
st.markdown("---")

# Main Area Upload
uploaded_file = st.file_uploader("Upload Lab Report (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Document", use_container_width=True)
        temp_path = "temp_img.png"
        img.save(temp_path)
    
    with col2:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("Extracting Data..."):
                data = run_pipeline(temp_path)
            
            if "error" in data:
                st.error(data["error"])
            else:
                # Display Results
                st.success("Extraction Successful!")
                
                # Metadata
                st.subheader("📋 Patient Info")
                # Logic to grab Patient/Doctor from FHIR Bundle
                patient_name = "N/A"
                for entry in data.get('entry', []):
                    res = entry.get('resource', {})
                    if res.get('resourceType') == 'Patient':
                        patient_name = res.get('name', [{}])[0].get('text', 'N/A')
                
                st.write(f"**Patient Name:** {patient_name}")
                
                # Observations Table
                obs = []
                for entry in data.get('entry', []):
                    res = entry.get('resource', {})
                    if res.get('resourceType') == 'Observation':
                        name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Test')
                        val = res.get('valueQuantity', {}).get('value', 'N/A')
                        unit = res.get('valueQuantity', {}).get('unit', '')
                        obs.append({"Test": name, "Result": f"{val} {unit}"})
                
                if obs:
                    st.table(pd.DataFrame(obs))
                
                st.download_button("📩 Download FHIR JSON", data=json.dumps(data, indent=2), file_name="record.json")