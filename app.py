import streamlit as st
import numpy as np
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Makanan Sehat",
    page_icon="🥗",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #2E7D32;
    }
    .subtitle {
        font-size: 16px;
        color: #AAAAAA;
    }
    .card {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .result-healthy {
        background-color: #1B5E20;
        padding: 18px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .result-unhealthy {
        background-color: #B71C1C;
        padding: 18px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
model = joblib.load("model_makanan.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="title">🥗 Sistem Klasifikasi Makanan Sehat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Prediksi makanan sehat & tidak sehat berbasis kandungan nutrisi menggunakan Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

# ======================================================
# SIDEBAR - INFORMASI MAKANAN
# ======================================================
st.sidebar.header("📋 Informasi Makanan")

nama_makanan = st.sidebar.text_input("Nama Makanan")

asal_masakan = st.sidebar.selectbox(
    "Asal Masakan",
    ["Indonesia", "Asia", "Eropa", "Amerika", "Timur Tengah"]
)

jenis_makanan = st.sidebar.selectbox(
    "Jenis Makanan",
    ["Makanan Utama", "Sarapan", "Cemilan", "Dinner", "Dessert"]
)

jenis_diet = st.sidebar.selectbox(
    "Jenis Diet",
    ["Umum", "Vegetarian", "Vegan", "Rendah Lemak", "Rendah Gula", "Tinggi Protein"]
)

metode_masak = st.sidebar.selectbox(
    "Metode Memasak",
    ["Rebus", "Kukus", "Panggang", "Tumis", "Goreng", "Tanpa Dimasak"]
)

waktu_persiapan = st.sidebar.number_input(
    "Waktu Persiapan (menit)", min_value=0, value=0, step=1, format="%d"
)

waktu_memasak = st.sidebar.number_input(
    "Waktu Memasak (menit)", min_value=0, value=0, step=1, format="%d"
)

rating = st.sidebar.slider("Rating Makanan", 1, 5, 3)

# ======================================================
# MAIN CONTENT
# ======================================================
col_left, col_right = st.columns([2, 1])

# =========================
# INPUT NUTRISI (INTEGER)
# =========================
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔢 Kandungan Nutrisi")

    c1, c2, c3 = st.columns(3)

    with c1:
        kalori = st.number_input("Kalori", min_value=0, value=0, step=1, format="%d")
        protein = st.number_input("Protein (g)", min_value=0, value=0, step=1, format="%d")
        serat = st.number_input("Serat (g)", min_value=0, value=0, step=1, format="%d")

    with c2:
        karbohidrat = st.number_input("Karbohidrat (g)", min_value=0, value=0, step=1, format="%d")
        lemak = st.number_input("Lemak (g)", min_value=0, value=0, step=1, format="%d")
        gula = st.number_input("Gula (g)", min_value=0, value=0, step=1, format="%d")

    with c3:
        natrium = st.number_input("Natrium (mg)", min_value=0, value=0, step=1, format="%d")
        kolesterol = st.number_input("Kolesterol (mg)", min_value=0, value=0, step=1, format="%d")
        porsi = st.number_input("Porsi Sajian (g)", min_value=0, value=0, step=1, format="%d")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDIKSI & RINGKASAN
# =========================
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediksi")

    if st.button("Prediksi Kesehatan", use_container_width=True):

        data_input = np.array([[
            float(kalori), float(protein), float(karbohidrat),
            float(lemak), float(serat), float(gula),
            float(natrium), float(kolesterol), float(porsi)
        ]])

        data_scaled = scaler.transform(data_input)
        hasil_prediksi = model.predict(data_scaled)

        if hasil_prediksi[0] == 1:
            st.markdown('<div class="result-healthy">✅ MAKANAN SEHAT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-unhealthy">❌ MAKANAN TIDAK SEHAT</div>', unsafe_allow_html=True)

        st.divider()
        st.caption("📄 Ringkasan")
        st.write(f"**Nama:** {nama_makanan if nama_makanan else '-'}")
        st.write(f"**Jenis:** {jenis_makanan}")
        st.write(f"**Diet:** {jenis_diet}")
        st.write(f"**Metode:** {metode_masak}")
        st.write(f"**Waktu Total:** {waktu_persiapan + waktu_memasak} menit")
        st.write(f"**Rating:** ⭐ {rating}/5")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "🔬 Model Machine Learning berbasis Random Forest / Gradient Boosting. "
    "Prediksi hanya menggunakan data nutrisi, atribut lain bersifat informatif."
)
