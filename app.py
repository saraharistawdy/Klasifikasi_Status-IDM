
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import os
import joblib

# Model selection
st.sidebar.subheader("Pilih Model")
model_option = st.sidebar.selectbox("Model:", ["Random Forest", "Gradient Boosting"])

# Title
st.title(f"Klasifikasi Status Indeks Desa Membangun di Jawa Tengah Algoritma Machine Learning: {model_option}")

# Layout & Logo
st.set_page_config(page_title="Klasifikasi Status Desa", page_icon="📊", layout="wide")

# Load and display logo
if os.path.exists("logo_undip.png"):
    logo = Image.open("logo_undip.png")
    st.image(logo, width=150)
else:
    st.warning("Logo tidak ditemukan")

# IDM and algorithm explanation section
with st.expander("📝 Penjelasan IDM dan Algoritma"):
    st.markdown("**Indeks Desa Membangun (IDM)** adalah indeks yang digunakan untuk mengukur tingkat perkembangan desa di Indonesia. IDM dibentuk dari tiga dimensi: **IKS**, **IKE**, dan **IKL**.")
    
    if model_option == "Random Forest":
        st.markdown("**🌳 Apa itu Random Forest?** Random Forest bekerja dengan membangun sejumlah pohon keputusan dan kemudian menggabungkan hasilnya untuk membuat prediksi akhir. Setiap pohon dilatih pada sampel acak dari data (bagging). Untuk klasifikasi, keluaran akhir ditentukan oleh suara mayoritas dari semua pohon.")
    else:
        st.markdown("**🚀 Apa itu Gradient Boosting?** Gradient Boosting membangun model prediktif yang kuat dengan menggabungkan banyak *weak learner* (pohon keputusan) secara berurutan. Setiap pohon baru dilatih untuk memperbaiki kesalahan (residual) yang dibuat oleh pohon-pohon sebelumnya.")

# Load model file
model_file = "rf_model_90.joblib" if model_option == "Random Forest" else "gbm_model_80.joblib"

if not os.path.exists(model_file):
    st.error(f"File model '{model_file}' tidak ditemukan!")
    st.stop()

try:
    model = joblib.load(model_file)
    st.sidebar.success(f"✅ Model {model_option} berhasil dimuat")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Manual input section
st.subheader("Input Manual")
col1, col2, col3 = st.columns(3)

iks = col1.number_input("IKS", 0.0, 1.0, 0.5, 0.01)
ike = col2.number_input("IKE", 0.0, 1.0, 0.5, 0.01)
ikl = col3.number_input("IKL", 0.0, 1.0, 0.5, 0.01)

# Button to classify manual input
if st.button("Klasifikasi Manual"):
    try:
        features = np.array([[iks, ike, ikl]])
        pred = model.predict(features)[0]
        idm_manual = np.mean(features)
        st.success(f"Kelas: {pred}")
        st.info(f"Nilai IDM: {idm_manual:.3f}")
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")

# CSV upload section for batch prediction
st.subheader("Upload CSV untuk Prediksi Status Desa")
uploaded_file = st.file_uploader("Pilih CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diupload:")
        st.dataframe(df.head())

        required_cols = ['iks_2024', 'ike_2024', 'ikl_2024']
        if all(col in df.columns for col in required_cols):
            # Calculate IDM value and make predictions
            df['status_idm_2024'] = df[required_cols].mean(axis=1).round(3)
            X = df[required_cols].values
            df['prediksi'] = model.predict(X)

            st.write("Hasil Prediksi + IDM:")
            st.dataframe(df)

            # Button to download CSV with prediction results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download Hasil Prediksi",
                data=csv,
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

            # Visualization of prediction distribution
            pie_data = df['prediksi'].value_counts().reset_index()
            pie_data.columns = ['Kelas', 'Jumlah']

            fig_pie = px.pie(pie_data, values='Jumlah', names='Kelas',
                             title="Distribusi Prediksi", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)

            fig_bar = px.bar(pie_data, x='Kelas', y='Jumlah', text='Jumlah', title="Jumlah per Kelas")
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            # Error handling for missing columns
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Kolom hilang: {missing}")

    except Exception as e:
        st.error(f"Error: {e}")
