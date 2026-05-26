import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0b1120, #111827, #1e293b);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Main Heading */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 10px;
    letter-spacing: 1px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 35px;
}

/* Glass Effect Container */
.section-box {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.12);

    padding: 28px;
    border-radius: 22px;

    margin-bottom: 25px;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);
}

/* Titles */
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 18px;
}

/* Normal Text */
.normal-text {
    font-size: 18px;
    color: #e2e8f0;
    line-height: 1.9;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 15px;
    border: 1px dashed #38bdf8;
}

/* Prediction Box */
.prediction-box {
    background: linear-gradient(
        135deg,
        rgba(16,185,129,0.25),
        rgba(5,150,105,0.35)
    );

    border: 1px solid rgba(16,185,129,0.4);

    padding: 25px;
    border-radius: 20px;

    text-align: center;

    margin-top: 20px;

    box-shadow: 0px 4px 18px rgba(0,0,0,0.3);
}

/* Recommendation Box */
.recommend-box {
    background: linear-gradient(
        135deg,
        rgba(239,68,68,0.18),
        rgba(127,29,29,0.35)
    );

    border: 1px solid rgba(239,68,68,0.3);

    padding: 24px;
    border-radius: 20px;

    margin-top: 20px;

    box-shadow: 0px 4px 18px rgba(0,0,0,0.3);
}

/* Graph Styling */
canvas {
    border-radius: 15px !important;
}

/* Image Styling */
img {
    border-radius: 18px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #38bdf8;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #0ea5e9, #2563eb);
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 12px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #0284c7, #1d4ed8);
}

/* Metric Style */
h2, h3 {
    color: #f8fafc;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================

st.markdown(
    """
    <div class='main-title'>
    🚧 AI-Based Road Damage Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
    Smart City Infrastructure Monitoring using CNN
    </div>
    """,
    unsafe_allow_html=True
)

# ================= ABOUT PROJECT =================

st.markdown("<div class='section-box'>", unsafe_allow_html=True)

st.markdown(
    "<div class='section-title'>📌 About the Project</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='normal-text'>

    • Road monitoring is important to prevent accidents and improve public safety.<br><br>

    • Damaged roads like potholes and cracks can increase vehicle damage and traffic risks.<br><br>

    • Convolutional Neural Networks (CNNs) help computers analyze and classify road surface images automatically.<br><br>

    • AI-based monitoring systems are widely used in smart cities, transportation systems, autonomous vehicles, and infrastructure maintenance.

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= MODEL =================

# Replace this with your trained model later
# Example:
# model = joblib.load("road_damage_model.pkl")

classes = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ================= IMAGE UPLOAD =================

st.markdown("<div class='section-box'>", unsafe_allow_html=True)

st.markdown(
    "<div class='section-title'>📤 Upload Road Image</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE PREVIEW =================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>🖼 Uploaded Image Preview</div>",
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= FAKE PREDICTION =================
    # Replace with actual CNN prediction later

    probabilities = np.random.dirichlet(np.ones(3), size=1)[0]

    pred_index = np.argmax(probabilities)

    pred_label = classes[pred_index]

    confidence = probabilities[pred_index] * 100

    # ================= SEVERITY =================

    if confidence > 85:
        severity = "High"
        sev_color = "#ef4444"

    elif confidence > 65:
        severity = "Medium"
        sev_color = "#f59e0b"

    else:
        severity = "Low"
        sev_color = "#22c55e"

    # ================= PREDICTION AREA =================

    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <h2>🚨 {pred_label} Detected</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        <h3 style='color:{sev_color};'>Severity: {severity}</h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUALIZATION =================

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>📊 Class Confidence Graph</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(7,4))

    bars = ax.bar(classes, probabilities)

    for bar in bars:
        bar.set_alpha(0.8)

    ax.set_ylabel("Confidence")

    ax.set_ylim([0,1])

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================

    st.markdown("<div class='recommend-box'>", unsafe_allow_html=True)

    st.markdown(
        "<h2>⚠ Recommendations</h2>",
        unsafe_allow_html=True
    )

    if severity == "High":

        st.markdown("""
        - Immediate maintenance recommended.<br>
        - High-risk road condition detected.<br>
        - Authorities should prioritize repair work.
        """, unsafe_allow_html=True)

    elif severity == "Medium":

        st.markdown("""
        - Schedule maintenance soon.<br>
        - Moderate road damage detected.<br>
        - Regular monitoring recommended.
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        - Minor road issue detected.<br>
        - Continue periodic inspections.<br>
        - Low immediate safety risk.
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
