import streamlit as st
import numpy as np
import joblib

# ======================================================
# KONFIGURASI HALAMAN Tes
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Makanan Sehat",
    page_icon="🥗",
    layout="centered"
)

# ======================================================
# LOAD MODEL & SCALER Tes
# ======================================================
model = joblib.load("model_makanan.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# JUDUL APLIKASI
# ======================================================
st.title("Sistem Klasifikasi Makanan Sehat")
st.write(
    "Sistem ini mengklasifikasikan makanan **sehat** dan **tidak sehat** "
    "berdasarkan **kandungan nutrisi** menggunakan model Machine Learning."
)

st.divider()

# ======================================================
# INFORMASI UMUM MAKANAN (UI SAJA)
# ======================================================
st.subheader("📝 Informasi Makanan")

nama_makanan = st.text_input("Nama Makanan")

asal_masakan = st.selectbox(
    "Asal Masakan",
    ["Indonesia", "Asia", "Eropa", "Amerika", "Timur Tengah"]
)

jenis_makanan = st.selectbox(
    "Jenis Makanan",
    ["Makanan Utama", "Sarapan", "Cemilan", "Dinner", "Dessert"]
)

jenis_diet = st.selectbox(
    "Jenis Diet",
    ["Umum", "Vegetarian", "Vegan", "Rendah Lemak", "Rendah Gula", "Tinggi Protein"]
)

metode_masak = st.selectbox(
    "Metode Memasak",
    ["Rebus", "Kukus", "Panggang", "Tumis", "Goreng", "Tanpa Dimasak"]
)

col_time1, col_time2 = st.columns(2)
with col_time1:
    waktu_persiapan = st.number_input("Waktu Persiapan (menit)", min_value=0)
with col_time2:
    waktu_memasak = st.number_input("Waktu Memasak (menit)", min_value=0)

rating = st.slider("Rating Makanan", 1, 5, 3)

st.divider()

# ======================================================
# INPUT NUTRISI (DIGUNAKAN MODEL)
# ======================================================
st.subheader("🔢 Masukkan Kandungan Nutrisi")

col1, col2 = st.columns(2)

with col1:
    kalori = st.number_input("Kalori", min_value=0.0)
    protein = st.number_input("Protein (gram)", min_value=0.0)
    karbohidrat = st.number_input("Karbohidrat (gram)", min_value=0.0)
    lemak = st.number_input("Lemak (gram)", min_value=0.0)

with col2:
    serat = st.number_input("Serat (gram)", min_value=0.0)
    gula = st.number_input("Gula (gram)", min_value=0.0)
    natrium = st.number_input("Natrium (mg)", min_value=0.0)
    kolesterol = st.number_input("Kolesterol (mg)", min_value=0.0)

porsi = st.number_input("Porsi Sajian (gram)", min_value=0.0)

st.divider()

# ======================================================
# PROSES PREDIKSI
# ======================================================
if st.button("🔍 Prediksi Kesehatan Makanan", use_container_width=True):

    data_input = np.array([[
        kalori, protein, karbohidrat, lemak,
        serat, gula, natrium, kolesterol, porsi
    ]])

    data_scaled = scaler.transform(data_input)
    hasil_prediksi = model.predict(data_scaled)

    st.subheader("📊 Hasil Prediksi")

    if hasil_prediksi[0] == 1:
        st.success("✅ Makanan ini diklasifikasikan sebagai **SEHAT**")
    else:
        st.error("❌ Makanan ini diklasifikasikan sebagai **TIDAK SEHAT**")

    # ==================================================
    # RINGKASAN INPUT (NILAI TAMBAH)
    # ==================================================
    st.divider()
    st.subheader("📋 Ringkasan Informasi")

    st.write(f"**Nama Makanan:** {nama_makanan if nama_makanan else '-'}")
    st.write(f"**Asal Masakan:** {asal_masakan}")
    st.write(f"**Jenis Makanan:** {jenis_makanan}")
    st.write(f"**Jenis Diet:** {jenis_diet}")
    st.write(f"**Metode Memasak:** {metode_masak}")
    st.write(f"**Waktu Total:** {waktu_persiapan + waktu_memasak} menit")
    st.write(f"**Rating:** ⭐ {rating}/5")

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "Model klasifikasi menggunakan Random Forest / Gradient Boosting "
    "berdasarkan kandungan nutrisi makanan. "
    "Atribut non-nutrisi hanya digunakan untuk tampilan."
)