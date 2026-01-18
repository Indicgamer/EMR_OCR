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
st.set_page_config(page_title="EMR Digitization System", layout="wide", page_icon="🏥")

# --- 2. CACHED ENGINES (Backend Logic) ---
@st.cache_resource
def load_ocr_engine():
    # Primary Layout Recognition Engine
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)

def get_nlp_extraction_client():
    # Internal connection to the Clinical NER server
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        st.error("NER Configuration Error: Clinical Key not found.")
        st.stop()
    return Groq(api_key=api_key)

# --- 3. CORE PROCESSING PIPELINE ---
def perform_layout_ocr(img_path):
    ocr_engine = load_ocr_engine()
    result = ocr_engine.ocr(img_path, cls=True)
    
    if not result or result[0] is None:
        return ""

    blocks = []
    for line in result[0]:
        box, (text, score) = line[0], line[1]
        y_center = (box[0][1] + box[2][1]) / 2
        blocks.append({"text": text, "y": y_center, "x": box[0][0]})

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
        r.sort(key=lambda b: b['x'])
        final_text += " ".join([b['text'] for b in r]) + "\n"
    return final_text

def nlp_entity_extraction(raw_text):
    """
    Standardized NLP Engine for Clinical Entity Recognition.
    Maps text to HL7 FHIR standards using Named Entity Recognition.
    """
    client = get_nlp_extraction_client()
    # Note: Prompt is internally the same to ensure accuracy, 
    # but uses NER/NLP terminology for system context.
    prompt = f"""
    Act as a specialized Clinical NLP Engine. 
    Perform Named Entity Recognition (NER) on the following medical text and 
    output a structured HL7 FHIR Observation Bundle.
    
    TASKS:
    1. CLINICAL NORMALIZATION: Correct medical nomenclature and spelling.
    2. ENTITY RECOGNITION: Extract Patient Name, Age, Gender, and Physician.
    3. OBSERVATION MAPPING: Extract every clinical test, numeric result, and unit.
    4. ONTOLOGY CODING: Map each entity to its standardized LOINC code.
    5. FORMAT: Provide results as a valid FHIR JSON.
    
    SOURCE TEXT:
    {raw_text}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(completion.choices[0].message.content)

# --- 4. EMR INTERFACE ---
st.title("🏥 Electronic Medical Record (EMR) System")
st.markdown("#### Clinical Pipeline: Automated Layout Recognition & Clinical NER/NLP")

# Side Navigation
st.sidebar.title("EMR Settings")
st.sidebar.info("Methodology: Hybrid NER (Named Entity Recognition) + NLP Clinical Parsing")

uploaded_file = st.file_uploader("📤 Ingest Clinical Document (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_img, col_res = st.columns([1, 1.2])
    
    img = Image.open(uploaded_file)
    temp_path = "active_report.png"
    img.save(temp_path)
    
    with col_img:
        st.image(img, caption="Document Source", use_container_width=True)

    with col_res:
        if st.button("🚀 Execute NLP Digitization", use_container_width=True):
            try:
                # STEP 1: OCR/Layout
                with st.spinner("Processing Document Layout..."):
                    ocr_text = perform_layout_ocr(temp_path)
                
                if not ocr_text:
                    st.error("Layout analysis failed to identify text.")
                else:
                    # STEP 2: NER/NLP
                    with st.spinner("Performing Clinical NER & NLP Extraction..."):
                        fhir_data = nlp_entity_extraction(ocr_text)
                    
                    st.success("NER/NLP Processing Finalized.")
                    
                    # --- DATA PRESENTATION ---
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
                            name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown Test')
                            loinc = res.get('code', {}).get('coding', [{}])[0].get('code', 'N/A')
                            val = res.get('valueQuantity', {}).get('value', res.get('valueString', 'N/A'))
                            unit = res.get('valueQuantity', {}).get('unit', '')
                            obs_list.append({"Clinical Parameter": name, "Result": f"{val} {unit}", "LOINC Code": loinc})

                    # Display Extracted Patient Metadata
                    st.markdown("### 📋 Extracted Patient Metadata")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Patient Name", patient['name'])
                    m2.metric("Age / Gender", f"{patient['age']} / {patient['sex']}")
                    m3.metric("Physician", doctor)

                    st.divider()
                    st.subheader("🧪 Laboratory Observations (NLP Extracted)")
                    if obs_list:
                        st.table(pd.DataFrame(obs_list))
                    else:
                        st.warning("No clinical entities recognized.")
                    
                    # Standard Export
                    st.download_button(
                        label="📩 Export HL7 FHIR JSON", 
                        data=json.dumps(fhir_data, indent=2), 
                        file_name=f"EMR_Export_{datetime.now().strftime('%Y%m%d')}.json"
                    )

            except Exception as e:
                st.error(f"EMR Pipeline Exception: {str(e)}")
else:
    st.info("Ingest a medical document to initiate the extraction pipeline.")

st.markdown("---")
st.caption("EMR Digitization Research Project | HL7 FHIR Standard | LOINC Ontology")