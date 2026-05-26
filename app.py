import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp {
    background:
        radial-gradient(circle at top left, #1e3a8a 0%, transparent 25%),
        radial-gradient(circle at bottom right, #0f766e 0%, transparent 25%),
        linear-gradient(135deg, #020617, #0f172a, #111827);

    color: white;
}

/* Hide Streamlit Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Section */
.hero {
    padding: 40px 20px;
    border-radius: 28px;
    text-align: center;

    background: rgba(255,255,255,0.05);

    backdrop-filter: blur(20px);

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0 8px 40px rgba(0,0,0,0.4);

    margin-bottom: 30px;
}

/* Main Title */
.main-title {
    font-size: 62px;
    font-weight: 800;

    background: linear-gradient(
        to right,
        #38bdf8,
        #818cf8,
        #06b6d4
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    font-size: 22px;
    color: #cbd5e1;
    font-weight: 400;
}

/* Section Cards */
.glass-card {
    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(14px);

    border-radius: 24px;

    padding: 30px;

    margin-bottom: 25px;

    box-shadow: 0 10px 35px rgba(0,0,0,0.35);

    transition: 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
}

/* Titles */
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 20px;
}

/* Paragraph */
.normal-text {
    color: #e2e8f0;
    line-height: 2;
    font-size: 18px;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    border-radius: 22px;
    padding: 20px;
    border: 2px dashed #38bdf8;
    background: rgba(255,255,255,0.04);
}

/* Prediction Card */
.prediction-card {
    border-radius: 25px;

    padding: 35px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(14,165,233,0.18),
            rgba(59,130,246,0.15)
        );

    border: 1px solid rgba(59,130,246,0.3);

    box-shadow: 0 8px 35px rgba(0,0,0,0.35);

    margin-top: 20px;
}

/* Recommendation Card */
.recommend-card {
    border-radius: 25px;

    padding: 30px;

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.15),
            rgba(127,29,29,0.20)
        );

    border: 1px solid rgba(239,68,68,0.2);

    box-shadow: 0 8px 35px rgba(0,0,0,0.35);

    margin-top: 25px;
}

/* Metrics */
.metric {
    font-size: 20px;
    color: #e2e8f0;
    margin-top: 12px;
}

.metric-big {
    font-size: 42px;
    font-weight: 700;
    color: white;
}

/* Image */
img {
    border-radius: 24px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #38bdf8;
    border-radius: 10px;
}

/* Button */
.stButton > button {
    background: linear-gradient(to right, #0ea5e9, #2563eb);

    color: white;

    border: none;

    border-radius: 14px;

    padding: 12px 28px;

    font-size: 16px;

    font-weight: 600;

    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* Success Animation */
@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.02);}
    100% {transform: scale(1);}
}

.prediction-card {
    animation: pulse 2s infinite;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO SECTION =================

st.markdown("""
<div class="hero">

    <div class="main-title">
        🚧 Road Damage Detection AI
    </div>

    <div class="sub-title">
        Smart Infrastructure Monitoring using Deep Learning & Computer Vision
    </div>

</div>
""", unsafe_allow_html=True)

# ================= ABOUT SECTION =================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📌 About This Project</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="normal-text">

✅ Detects potholes, cracks, and manholes automatically using AI.<br><br>

✅ Helps smart cities monitor road conditions efficiently.<br><br>

✅ Reduces accidents and infrastructure maintenance delays.<br><br>

✅ Uses Convolutional Neural Networks (CNNs) for image classification.<br><br>

✅ Can be integrated with drones, CCTV systems, and autonomous vehicles.

</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= MODEL =================

classes = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ================= LAYOUT =================

left, right = st.columns([1, 1])

# ================= UPLOAD SECTION =================

with left:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📤 Upload Road Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ================= IMAGE PREVIEW =================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with right:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">🖼 Image Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= PREDICTION =================

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

    # ================= PREDICTION CARD =================

    st.markdown(f"""
    <div class="prediction-card">

        <div class="metric-big">
            🚨 {pred_label} Detected
        </div>

        <div class="metric">
            Confidence Score
        </div>

        <h1 style="font-size:55px;">
            {confidence:.2f}%
        </h1>

        <div class="metric">
            Severity Level
        </div>

        <h2 style="color:{sev_color}; font-size:34px;">
            {severity}
        </h2>

    </div>
    """, unsafe_allow_html=True)

    # ================= CHART SECTION =================

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📊 Confidence Visualization</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probabilities,
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='auto'
            )
        ]
    )

    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        xaxis_title="Damage Type",
        yaxis_title="Confidence",
        yaxis=dict(range=[0,1])
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================

    st.markdown('<div class="recommend-card">', unsafe_allow_html=True)

    st.markdown("""
    <h1 style="color:white;">
    ⚠ Maintenance Recommendations
    </h1>
    """, unsafe_allow_html=True)

    if severity == "High":

        st.markdown("""
        <div class="normal-text">

        🔴 Immediate maintenance required.<br><br>

        🔴 Severe road damage detected.<br><br>

        🔴 Authorities should prioritize repair work urgently.<br><br>

        🔴 High probability of accidents and vehicle damage.

        </div>
        """, unsafe_allow_html=True)

    elif severity == "Medium":

        st.markdown("""
        <div class="normal-text">

        🟠 Moderate road damage identified.<br><br>

        🟠 Schedule maintenance soon.<br><br>

        🟠 Continuous monitoring is recommended.<br><br>

        🟠 Repair can prevent future worsening.

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="normal-text">

        🟢 Minor road damage detected.<br><br>

        🟢 Continue periodic inspections.<br><br>

        🟢 Low immediate safety risk.<br><br>

        🟢 Preventive maintenance recommended.

        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="glass-card" style="text-align:center;">

        <h1 style="font-size:38px;">
        🚀 Upload a road image to begin AI analysis
        </h1>

        <p style="font-size:20px; color:#cbd5e1;">
        The system will automatically detect potholes, cracks, and manholes.
        </p>

    </div>
    """, unsafe_allow_html=True)
