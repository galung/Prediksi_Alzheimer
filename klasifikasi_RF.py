import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load Model, Scaler ===
model = joblib.load("model_random_forest_alzheimer.pkl")
scaler = joblib.load("scaler.pkl")

# === Label Encoding ===
kategori = {
    'Gender': {'Male': 0, 'Female': 1},
    'Ethnicity': {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3},
    'EducationLevel': {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3},
    'Smoking': {'No': 0, 'Yes': 1},
    'FamilyHistoryAlzheimers': {'No': 0, 'Yes': 1},
    'CardiovascularDisease': {'No': 0, 'Yes': 1},
    'Diabetes': {'No': 0, 'Yes': 1},
    'Depression': {'No': 0, 'Yes': 1},
    'HeadInjury': {'No': 0, 'Yes': 1},
    'Hypertension': {'No': 0, 'Yes': 1},
    'MemoryComplaints': {'No': 0, 'Yes': 1},
    'BehavioralProblems': {'No': 0, 'Yes': 1},
    'Confusion': {'No': 0, 'Yes': 1},
    'Disorientation': {'No': 0, 'Yes': 1},
    'PersonalityChanges': {'No': 0, 'Yes': 1},
    'DifficultyCompletingTasks': {'No': 0, 'Yes': 1},
    'Forgetfulness': {'No': 0, 'Yes': 1},
}

# === Fitur Urut Sesuai Training ===
feature_order = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
    'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
    'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness'
]

numerik_cols = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
    'FunctionalAssessment', 'ADL'
]

placeholders = {
    'Age': "Masukkan Umur",
    'BMI': "Masukkan nilai BMI (15–40)",
    'AlcoholConsumption': "Masukkan konsumsi alkohol per minggu",
    'PhysicalActivity': "Masukkan frekuensi aktivitas fisik per minggu",
    'DietQuality': "Masukkan skor kualitas diet (1–10)",
    'SleepQuality': "Masukkan skor kualitas tidur (1–10)",
    'SystolicBP': "Masukkan tekanan darah sistolik (mmHg)",
    'DiastolicBP': "Masukkan tekanan darah diastolik (mmHg)",
    'CholesterolTotal': "Masukkan total kolesterol (mg/dL)",
    'CholesterolLDL': "Masukkan LDL (mg/dL)",
    'CholesterolHDL': "Masukkan HDL (mg/dL)",
    'CholesterolTriglycerides': "Masukkan trigliserida (mg/dL)",
    'MMSE': "Masukkan skor MMSE (0–30)",
    'FunctionalAssessment': "Masukkan skor fungsional",
    'ADL': "Masukkan skor aktivitas harian"
}

# === Streamlit UI ===
st.title("🧠 Prediksi Risiko Alzheimer")
st.markdown("Silakan masukkan data pasien untuk memprediksi risiko Alzheimer:")

user_input = {}
for feature in feature_order:
    if feature in kategori:
        user_input[feature] = st.selectbox(f"{feature}", options=list(kategori[feature].keys()))
    else:
        placeholder_text = placeholders.get(feature, f"Masukkan nilai {feature}")
        user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1, placeholder=placeholder_text)

        if feature == 'BMI':
            st.caption("💡 BMI = Berat Badan / (Tinggi Badan)^2. Normal: 15–40")
        elif feature == 'AlcoholConsumption':
            st.caption("💡 0 = Tidak Minum, 1–20 = Frekuensi/pekan")
        elif feature == 'DietQuality':
            st.caption("💡 Skor diet bisa dihitung dari: [Link Kalkulator Diet](https://www.dietquality.org/calculator)")

if st.button("🔍 Prediksi Sekarang"):
    try:
        # Encode input
        input_data = []
        for feat in feature_order:
            val = user_input[feat]
            if feat in kategori:
                val = kategori[feat][val]
            input_data.append(val)

        # Buat DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_order)
        st.write("📊 Data Input Sebelum Scaling:")
        st.dataframe(input_df)

        # Normalisasi kolom numerik
        input_df[numerik_cols] = scaler.transform(input_df[numerik_cols])
        st.write("🔍 Data Setelah Scaling:")
        st.dataframe(input_df)

        # Validasi kesesuaian fitur
        if hasattr(model, 'feature_names_in_'):
            missing = set(model.feature_names_in_) - set(input_df.columns)
            if missing:
                st.error(f"❌ Kolom hilang pada input: {missing}")
                st.stop()

        # Prediksi
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.write(f"🔢 Probabilitas: Tidak Alzheimer = {proba[0]:.4f}, Alzheimer = {proba[1]:.4f}")
        if pred == 1:
            st.error("❌ Pasien Diduga Mengalami Alzheimer")
        else:
            st.success("✅ Pasien Tidak Mengalami Alzheimer")

    except Exception as e:
        st.error(f"🚨 Terjadi kesalahan saat memproses data: {e}")
