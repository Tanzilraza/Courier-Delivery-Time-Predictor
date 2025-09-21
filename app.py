# ---------------------------------
# Courier Delivery Time Predictor.
# ---------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import joblib
import time
import threading
import pyttsx3
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(page_title="🚚 Courier Time Predictor", layout="centered")
st.markdown("""
    <h1 style='text-align:center;color:#4B8BBE;'>🚀 Courier Delivery Time Predictor</h1>
    <p style='text-align:center;'>Predict courier delivery time using ML based on real-time inputs 📦</p>
""", unsafe_allow_html=True)

# -----------------------------------------------
# TRAINING FAKE DATA & MODEL
# -----------------------------------------------
@st.cache_data
def train_model():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "Distance": np.random.uniform(1, 1000, n),
        "Weight": np.random.uniform(0.1, 50, n),
        "Traffic": np.random.choice(["Low", "Medium", "High"], n),
        "Area": np.random.choice(["Urban", "Rural"], n),
        "Weather": np.random.choice(["Clear", "Rainy", "Stormy"], n)
    })
    def calculate_time(row):
        base = row['Distance'] / 40 + row['Weight'] * 0.1
        base += {"Low":0, "Medium":1.5, "High":3}[row['Traffic']]
        base += {"Urban":0, "Rural":2}[row['Area']]
        base += {"Clear":0, "Rainy":1.5, "Stormy":3}[row['Weather']]
        return round(base + np.random.normal(0, 0.5), 2)

    df['DeliveryTime'] = df.apply(calculate_time, axis=1)
    df_enc = df.replace({"Traffic":{"Low":0, "Medium":1, "High":2},
                         "Area":{"Urban":0, "Rural":1},
                         "Weather":{"Clear":0, "Rainy":1, "Stormy":2}})

    X = df_enc.drop("DeliveryTime", axis=1)
    y = df_enc["DeliveryTime"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, df, mae, r2

model, data, mae, r2 = train_model()

# -----------------------------------------------
# USER INPUT FORM
# -----------------------------------------------
st.subheader("📥 Enter Courier Details")
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        distance = st.slider("📏 Distance (km)", 1.0, 1000.0, 100.0)
    with c2:
        weight = st.slider("⚖️ Weight (kg)", 0.1, 50.0, 5.0)
    with c3:
        traffic = st.selectbox("🚦 Traffic", ["Low", "Medium", "High"])

    c4, c5 = st.columns(2)
    with c4:
        area = st.selectbox("🏙️ Area", ["Urban", "Rural"])
    with c5:
        weather = st.selectbox("☁️ Weather", ["Clear", "Rainy", "Stormy"])

    submit = st.form_submit_button("🔮 Predict Now")

# -----------------------------------------------
# HELPER: PDF GENERATION
# -----------------------------------------------
def generate_pdf(details, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, "Courier Delivery Prediction Result", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    for key, val in details.items():
        pdf.cell(190, 10, f"{key}: {val}", 0, 1)
    pdf.cell(190, 10, f"Estimated Delivery Time: {prediction} days", 0, 1)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buffer = BytesIO(pdf_bytes)
    return buffer

# -----------------------------------------------
# HELPER: SAFE VOICE
# -----------------------------------------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        pass

# -----------------------------------------------
# PREDICTION
# -----------------------------------------------
if submit:
    with st.spinner("Predicting delivery time... 📦"):
        time.sleep(1.2)
        input_data = pd.DataFrame({
            "Distance": [distance],
            "Weight": [weight],
            "Traffic": [{"Low":0,"Medium":1,"High":2}[traffic]],
            "Area": [{"Urban":0,"Rural":1}[area]],
            "Weather": [{"Clear":0,"Rainy":1,"Stormy":2}[weather]]
        })
        pred = round(model.predict(input_data)[0], 2)
        status = "🟢 Fast" if pred < 2 else "🟡 Normal" if pred < 5 else "🔴 Delayed"
        bg = "#d1e7dd" if "🟢" in status else "#fff3cd" if "🟡" in status else "#f8d7da"

        st.markdown(f"""
            <div style='background-color:{bg}; padding:20px; border-radius:10px;'>
                <h3>📦 Estimated Delivery Time: {pred} days ({status})</h3>
            </div>
        """, unsafe_allow_html=True)

        # 🔊 Voice Output in Thread
        threading.Thread(target=speak, args=(f"Estimated delivery time is {pred} days",)).start()

        # 💬 Smart Tip
        if traffic == "High" and weather == "Stormy":
            st.error("⚠️ Heavy delay expected due to traffic and stormy weather!")
        elif area == "Rural" and weather != "Clear":
            st.warning("⏳ Rural deliveries take longer in bad weather.")

        # 📜 History
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({
            "Distance (km)": distance,
            "Weight (kg)": weight,
            "Traffic": traffic,
            "Area": area,
            "Weather": weather,
            "Prediction (days)": pred
        })

        # 📥 Download PDF
        pdf_file = generate_pdf(st.session_state["history"][-1], pred)
        st.download_button("📥 Download PDF Slip", pdf_file.getvalue(), file_name="delivery_prediction.pdf")

# -----------------------------------------------
# HISTORY TABLE
# -----------------------------------------------
if "history" in st.session_state and st.session_state["history"]:
    st.subheader("📜 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state["history"]))

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
st.subheader("📊 Data Insights")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=data, x="Distance", y="DeliveryTime", hue="Area", ax=ax1)
ax1.set_title("Delivery Time vs Distance by Area")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.boxplot(data=data, x="Traffic", y="DeliveryTime", ax=ax2)
ax2.set_title("Traffic Effect on Delivery Time")
st.pyplot(fig2)

# -----------------------------------------------
# MODEL METRICS
# -----------------------------------------------
st.subheader("📈 Model Performance")
st.success(f"✅ Mean Absolute Error: {mae:.2f} days")
st.success(f"📉 R² Score: {r2:.2f}")

buffer = BytesIO()
joblib.dump(model, buffer)
st.download_button("💾 Download Trained Model", buffer.getvalue(), file_name="trained_model.joblib")

