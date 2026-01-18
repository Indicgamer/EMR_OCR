import streamlit as st
import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCR
from groq import Groq
from datetime import datetime

# --- 1. INITIAL SETTINGS ---
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
st.set_page_config(page_title="EMR AI Digitizer", layout="wide", page_icon="🏥")

# --- 2. CACHED ENGINES (Load Once) ---
@st.cache_resource
def load_ocr_engine():
    # Initializes PaddleOCR once and keeps it in memory
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)

def get_llm_client():
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in Colab Secrets.")
        st.stop()
    return Groq(api_key=api_key)

# --- 3. CORE LOGIC FUNCTIONS ---
def perform_ocr(img_path):
    ocr_engine = load_ocr_engine()
    result = ocr_engine.ocr(img_path, cls=True)
    
    if not result or result[0] is None:
        return ""

    blocks = []
    for line in result[0]:
        box, (text, score) = line[0], line[1]
        # Calculate center Y to group text into rows
        y_center = (box[0][1] + box[2][1]) / 2
        blocks.append({"text": text, "y": y_center, "x": box[0][0]})

    # Row-grouping logic (15px threshold)
    blocks.sort(key=lambda b: b['y'])
    rows, current_row = [], [blocks[0]]
    for i in range(1, len(blocks)):
        if abs(blocks[i]['y'] - blocks[i-1]['y']) < 15:
            current_row.append(blocks[i])
        else:
            rows.append(current_row)
            current_row = [blocks[i]]
    rows.append(current_row)

    final_text = ""
    for r in rows:
        r.sort(key=lambda b: b['x']) # Sort left-to-right
        final_text += " ".join([b['text'] for b in r]) + "\n"
    return final_text

def parse_with_llm(raw_text):
    client = get_llm_client()
    prompt = f"""
    Act as a Medical Information Extraction System. 
    Convert the following noisy OCR text into a structured HL7 FHIR Observation Bundle.
    
    TASKS:
    1. HEAL SPELLING: Correct clinical typos.
    2. EXTRACT METADATA: Patient Name, Age, Gender, and Doctor Name.
    3. EXTRACT RESULTS: Every test, numeric result, and unit. Ignore reference ranges.
    4. CODE: Map each test to its standard LOINC code.
    5. FORMAT: Valid JSON FHIR Bundle only.
    
    OCR TEXT:
    {raw_text}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(completion.choices[0].message.content)

# --- 4. STREAMLIT UI ---
st.title("🏥 Smart EMR OCR Digitizer")
st.markdown("#### MTech Project: Automated Clinical Pipeline (Stage 8 & 9)")

uploaded_file = st.file_uploader("📤 Upload Clinical Report (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_img, col_res = st.columns([1, 1.2])
    
    # Save temp image for PaddleOCR
    img = Image.open(uploaded_file)
    temp_path = "active_report.png"
    img.save(temp_path)
    
    with col_img:
        st.image(img, caption="Original Document", use_container_width=True)

    with col_res:
        if st.button("🚀 Start Digitization Pipeline", use_container_width=True):
            try:
                # STEP 1: OCR
                with st.spinner("Step 1: Extracting Text Layout..."):
                    ocr_text = perform_ocr(temp_path)
                
                if not ocr_text:
                    st.error("OCR failed to find text.")
                else:
                    # STEP 2: LLM
                    with st.spinner("Step 2: Semantic Analysis (Llama-3)..."):
                        fhir_data = parse_with_llm(ocr_text)
                    
                    st.success("Analysis Complete!")
                    
                    # --- RENDER RESULTS ---
                    patient = {"name": "N/A", "age": "N/A", "sex": "N/A"}
                    doctor = "N/A"
                    obs_list = []

                    for entry in fhir_data.get('entry', []):
                        res = entry.get('resource', {})
                        rtype = res.get('resourceType')
                        if rtype == 'Patient':
                            patient['name'] = res.get('name', [{}])[0].get('text', 'N/A')
                            patient['sex'] = res.get('gender', 'N/A').capitalize()
                            patient['age'] = res.get('birthDate') or "N/A"
                        elif rtype == 'Practitioner':
                            doctor = res.get('name', [{}])[0].get('text', 'N/A')
                        elif rtype == 'Observation':
                            name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
                            loinc = res.get('code', {}).get('coding', [{}])[0].get('code', 'N/A')
                            val = res.get('valueQuantity', {}).get('value', res.get('valueString', 'N/A'))
                            unit = res.get('valueQuantity', {}).get('unit', '')
                            obs_list.append({"Test": name, "Result": f"{val} {unit}", "LOINC": loinc})

                    # Display Patient Info
                    st.markdown("### 📋 Patient Metadata")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Name", patient['name'])
                    m2.metric("Age/Sex", f"{patient['age']} / {patient['sex']}")
                    m3.metric("Doctor", doctor)

                    st.divider()
                    st.subheader("🧪 Laboratory Observations")
                    if obs_list:
                        st.table(pd.DataFrame(obs_list))
                    
                    st.download_button("📩 Export FHIR JSON", data=json.dumps(fhir_data, indent=2), file_name="EMR_Record.json")

            except Exception as e:
                st.error(f"Pipeline Error: {str(e)}")
else:
    st.info("Please upload a lab report to begin processing.")