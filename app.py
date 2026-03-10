import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model yang sudah kamu simpan sebelumnya
with open('xgboost_diamond_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

st.title("💎 Diamond Price Prediction")
st.write("Aplikasi ini memprediksi harga berlian menggunakan model **XGBoost** dengan pembagian data **90:10**.")

# Input Form
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat Weight", min_value=0.1, max_value=5.0, value=0.7, step=0.01)
    cut = st.selectbox("Cut Quality", options=[0, 1, 2, 3, 4], 
                       format_func=lambda x: ["Fair", "Good", "Very Good", "Premium", "Ideal"][x])
    color = st.selectbox("Color", options=[0, 1, 2, 3, 4, 5, 6], 
                         format_func=lambda x: ["D (Best)", "E", "F", "G", "H", "I", "J (Worst)"][x])

with col2:
    clarity = st.selectbox("Clarity", options=[0, 1, 2, 3, 4, 5, 6, 7], 
                           format_func=lambda x: ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"][x])
    depth = st.number_input("Total Depth %", min_value=40.0, max_value=80.0, value=61.0)
    table = st.number_input("Table Width", min_value=40.0, max_value=100.0, value=57.0)

st.subheader("Dimensions")
c3, c4, c5 = st.columns(3)
with c3:
    x = st.number_input("Length (x) mm", min_value=0.0, value=5.7)
with c4:
    y = st.number_input("Width (y) mm", min_value=0.0, value=5.7)
with c5:
    z = st.number_input("Depth (z) mm", min_value=0.0, value=3.5)

st.divider()

# Tombol Prediksi
if st.button("Predict Price", use_container_width=True):
    # Susun data input sesuai urutan fitur saat training
    input_features = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])
    
    # Jalankan Prediksi
    prediction = model.predict(input_features)
    
    # Tampilkan Hasil Prediksi Utama
    st.success(f"### Predicted Price: ${prediction[0]:,.2f}")
    
    # Tambahkan Informasi Akurasi Model
    st.info("📊 **Model Performance Details (Split 90:10):**")
    acc_col1, acc_col2 = st.columns(2)
    
    with acc_col1:
        st.metric(label="R2 Score (Accuracy)", value="98.41%")
    with acc_col2:
        st.metric(label="Mean Absolute Error (MAE)", value="$195.79")

    st.caption("Catatan: Akurasi di atas didapat dari hasil pengujian model XGBoost pada data testing.")
