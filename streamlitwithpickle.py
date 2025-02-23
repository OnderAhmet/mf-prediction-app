import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Sayfa ayarları
st.set_page_config(page_title="Mf, Pf, and Pm Prediction", layout="wide")

# Stil ekleme (CSS ile düzenlenmiş şık bir görünüm)
st.markdown("""
    <style>
        .stTextInput, .stNumberInput {
            border-radius: 10px;
            border: 2px solid #4CAF50;
            padding: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px;
        }
        .result-box {
            background-color: #F0F0F0;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Başlık ve açıklama
st.title("🌍 AI-Powered Mf, Pf, and Pm Prediction")
st.markdown("### 📊 Enter the input values below and get real-time predictions.")


# Pickle dosyalarını yükle
@st.cache_resource
def load_models():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("gb_model.pkl", "rb") as f:
        gb_model = pickle.load(f)
    with open("catboost_model.pkl", "rb") as f:
        catboost_model = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    return preprocessor, gb_model, catboost_model, xgb_model


preprocessor, gb_model, catboost_model, xgb_model = load_models()

# Girdi değişkenleri
numeric_columns = [
    'Mud Weight, ppg', 'Yield Point, lbf/100 sqft', 'Chlorides, mg/L',
    'Solids, %vol', 'HTHP Fluid Loss, cc/30min', 'pH',
    'NaCl (SA), %vol', 'KCl (SA), %vol', 'Low Gravity (SA), %vol',
    'Drill Solids (SA), %vol', 'R600', 'R300', 'R200', 'R100', 'R6', 'R3', 'Average SG Solids (SA)'
]

# Kullanıcı giriş alanları (Grid şeklinde dizayn edilmiş)
input_data = []
col1, col2, col3 = st.columns(3)
for i, col in enumerate(numeric_columns):
    if i % 3 == 0:
        with col1:
            value = st.number_input(f'{col}', value=10.0)
    elif i % 3 == 1:
        with col2:
            value = st.number_input(f'{col}', value=10.0)
    else:
        with col3:
            value = st.number_input(f'{col}', value=10.0)

    input_data.append(value)

# Tahmin Butonu
if st.button("🔍 Predict Mf, Pf and Pm"):
    input_array = np.array(input_data).reshape(1, -1)

    # Veriyi dönüştürme
    input_transformed = preprocessor.transform(input_array)

    # Mf tahminini yapma
    mf_prediction = gb_model.predict(input_transformed)[0]

    # Pf ve Pm tahminlerini yapma
    mf_array = np.array([[mf_prediction]])
    pf_prediction = catboost_model.predict(mf_array)[0]
    pm_prediction = xgb_model.predict(mf_array)[0]

    # Sonuçları Şık Kutucuklar İçinde Gösterme
    st.markdown("## 🔮 Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='result-box'>Predicted Mf: {mf_prediction:.2f}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='result-box'>Predicted Pf: {pf_prediction:.2f}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='result-box'>Predicted Pm: {pm_prediction:.2f}</div>", unsafe_allow_html=True)
