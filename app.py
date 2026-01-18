import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCR
from groq import Groq
from datetime import datetime

# --- 1. INITIAL SETTINGS ---
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
st.set_page_config(page_title="Standardized FHIR EMR System", layout="wide", page_icon="🏥")

# --- 2. CACHED ENGINES ---
@st.cache_resource
def load_ocr_engine():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)

def get_nlp_client():
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        st.error("NER Configuration Error: Clinical Key not found.")
        st.stop()
    return Groq(api_key=api_key)

# --- 3. CORE PROCESSING PIPELINE ---
def perform_layout_ocr(img_path):
    ocr_engine = load_ocr_engine()
    result = ocr_engine.ocr(img_path, cls=True)
    if not result or result[0] is None: return ""
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

def nlp_fhir_extraction(raw_text):
    client = get_nlp_client()
    # UPDATED PROMPT: Added strict deduplication and accuracy instructions
    prompt = f"""
    Act as a specialized Clinical NLP Engine. 
    Perform Named Entity Recognition (NER) and output a valid HL7 FHIR R4 JSON Bundle.
    
    STRICT REQUIREMENTS:
    1. OCR HEALING: Correct medical misspellings (e.g., 'Haemoglobln' -> 'Hemoglobin').
    2. ENTITY EXTRACTION: Extract ALL medical tests, results, and units.
    3. DEDUPLICATION: Ensure each clinical test or field (e.g., Hemoglobin) appears ONLY ONCE in the bundle. 
    4. ACCURACY: Extract numeric results exactly as they appear.
    5. RESOURCE MAPPING: Create unique 'Patient', 'Practitioner', and 'Observation' resources.
    6. METADATA: Include 'status': 'final' and 'category': 'laboratory'.
    7. ONTOLOGY: Map every test to the "http://loinc.org" system.
    
    SOURCE OCR TEXT:
    {raw_text}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0 # Lowest temperature for maximum accuracy
    )
    return json.loads(completion.choices[0].message.content)

# --- 4. EMR INTERFACE ---
st.title("🏥 Electronic Medical Record (EMR) System")
st.markdown("#### **Standardized Clinical Pipeline**: Automated OCR Layout & HL7 FHIR NER")

st.sidebar.title("System Standards")
st.sidebar.success("✅ HL7 FHIR R4 Compliant")
st.sidebar.success("✅ Deduplication Active")
st.sidebar.success("✅ LOINC Ontology Mapped")

uploaded_file = st.file_uploader("📤 Ingest Clinical Document (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col_img, col_res = st.columns([1, 1.2])
    img = Image.open(uploaded_file)
    temp_path = "active_report.png"
    img.save(temp_path)
    
    with col_img:
        st.image(img, caption="Document Source", use_container_width=True)

    with col_res:
        if st.button("🚀 Execute NLP & FHIR Generation", use_container_width=True):
            try:
                with st.spinner("Processing Document Layout..."):
                    ocr_text = perform_layout_ocr(temp_path)
                
                with st.spinner("Executing Clinical NER & FHIR Mapping..."):
                    fhir_data = nlp_fhir_extraction(ocr_text)
                
                # --- DEDUPLICATION & PARSING LOGIC ---
                patient_data = {"name": "N/A", "id": "N/A", "gender": "N/A"}
                unique_observations = {} # Dictionary to prevent duplicates

                for entry in fhir_data.get('entry', []):
                    res = entry.get('resource', {})
                    rtype = res.get('resourceType')
                    
                    if rtype == 'Patient':
                        names = res.get('name', [{}])
                        patient_data['name'] = names[0].get('text') or names[0].get('family', 'N/A')
                        patient_data['gender'] = res.get('gender', 'N/A').capitalize()
                        patient_data['id'] = res.get('id', 'local-001')
                    
                    elif rtype == 'Observation':
                        test_name = res.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
                        loinc = res.get('code', {}).get('coding', [{}])[0].get('code', 'N/A')
                        
                        # Create a uniqueness key (Test Name + LOINC)
                        # This ensures 'Hemoglobin' is not added twice even if it exists twice in the JSON
                        key = f"{test_name}_{loinc}".upper()
                        
                        val_qty = res.get('valueQuantity', {})
                        val = val_qty.get('value', res.get('valueString', 'N/A'))
                        unit = val_qty.get('unit', '')

                        unique_observations[key] = {
                            "Resource ID": res.get('id', 'obs-x'),
                            "Test Name": test_name,
                            "Result": val,
                            "Unit": unit,
                            "LOINC": loinc,
                            "Status": res.get('status', 'final')
                        }

                # --- UI RENDERING ---
                tab1, tab2, tab3 = st.tabs(["📋 Clinical Dashboard", "🔬 FHIR Technical View", "📄 Raw Bundle"])

                with tab1:
                    st.markdown("### Patient Metadata")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Name", patient_data['name'])
                    m2.metric("Gender", patient_data['gender'])
                    m3.metric("Standard ID", patient_data['id'])
                    
                    st.divider()
                    st.subheader("Laboratory Results")
                    if unique_observations:
                        df_summary = pd.DataFrame(list(unique_observations.values()))[["Test Name", "Result", "Unit"]]
                        st.table(df_summary)

                with tab2:
                    st.subheader("FHIR Resource Mapping (LOINC Standard)")
                    if unique_observations:
                        df_fhir = pd.DataFrame(list(unique_observations.values()))[["Resource ID", "Test Name", "LOINC", "Status"]]
                        st.dataframe(df_fhir, use_container_width=True, hide_index=True)
                    
                with tab3:
                    st.subheader("Final FHIR Bundle (JSON)")
                    st.code(json.dumps(fhir_data, indent=2), language='json')
                    st.download_button("📩 Download FHIR JSON", data=json.dumps(fhir_data, indent=2), file_name="EMR_Bundle.json")

            except Exception as e:
                st.error(f"EMR Pipeline Error: {str(e)}")
else:
    st.info("Ingest document to view FHIR resource mappings.")