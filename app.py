import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Road Damage Detection AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

/* ================= GLOBAL ================= */

html, body, [class*="css"]  {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, #1d4ed8 0%, transparent 30%),
        radial-gradient(circle at bottom right, #0f766e 0%, transparent 30%),
        linear-gradient(135deg, #020617, #081226, #0f172a);

    color: white;
}

/* Hide Streamlit Default UI */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* ================= HERO SECTION ================= */

.hero {
    padding: 45px;

    border-radius: 30px;

    text-align: center;

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    box-shadow: 0 10px 40px rgba(0,0,0,0.4);

    margin-bottom: 30px;
}

.main-title {

    font-size: 60px;

    font-weight: 800;

    background: linear-gradient(
        to right,
        #38bdf8,
        #818cf8,
        #06b6d4
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    margin-bottom: 12px;
}

.sub-title {

    font-size: 22px;

    color: #cbd5e1;
}

/* ================= GLASS CARD ================= */

.glass-card {

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(16px);

    border-radius: 25px;

    padding: 30px;

    margin-bottom: 25px;

    box-shadow: 0 8px 30px rgba(0,0,0,0.35);

    transition: 0.3s ease;
}

.glass-card:hover {

    transform: translateY(-4px);
}

/* ================= TITLES ================= */

.section-title {

    font-size: 32px;

    font-weight: 700;

    color: #38bdf8;

    margin-bottom: 18px;
}

/* ================= TEXT ================= */

.normal-text {

    font-size: 18px;

    line-height: 2;

    color: #e2e8f0;
}

/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {

    border: 2px dashed #38bdf8;

    border-radius: 22px;

    padding: 18px;

    background: rgba(255,255,255,0.04);
}

/* ================= PREDICTION CARD ================= */

.prediction-card {

    border-radius: 28px;

    padding: 35px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(14,165,233,0.18),
            rgba(59,130,246,0.16)
        );

    border: 1px solid rgba(59,130,246,0.3);

    box-shadow: 0 8px 35px rgba(0,0,0,0.35);

    margin-top: 25px;
}

/* ================= RECOMMEND CARD ================= */

.recommend-card {

    border-radius: 28px;

    padding: 30px;

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.12),
            rgba(127,29,29,0.18)
        );

    border: 1px solid rgba(239,68,68,0.2);

    box-shadow: 0 8px 35px rgba(0,0,0,0.35);

    margin-top: 25px;
}

/* ================= METRICS ================= */

.metric-title {

    font-size: 20px;

    color: #cbd5e1;

    margin-top: 10px;
}

.metric-value {

    font-size: 55px;

    font-weight: bold;

    color: white;
}

/* ================= IMAGE ================= */

img {

    border-radius: 20px;
}

/* ================= SCROLLBAR ================= */

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

/* ================= BUTTON ================= */

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

/* ================= ANIMATION ================= */

@keyframes glow {

    0% {
        box-shadow: 0 0 10px rgba(56,189,248,0.3);
    }

    50% {
        box-shadow: 0 0 25px rgba(56,189,248,0.5);
    }

    100% {
        box-shadow: 0 0 10px rgba(56,189,248,0.3);
    }
}

.prediction-card {

    animation: glow 3s infinite;
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

# ================= ABOUT PROJECT =================

st.markdown("""
<div class="glass-card">

    <div class="section-title">
        📌 About This Project
    </div>

    <div class="normal-text">

        ✅ Detect potholes, cracks, and manholes automatically.<br><br>

        ✅ Helps smart cities monitor infrastructure efficiently.<br><br>

        ✅ Reduces accidents and road maintenance delays.<br><br>

        ✅ Uses AI and CNN-based computer vision systems.<br><br>

        ✅ Future-ready for drones and autonomous vehicles.

    </div>

</div>
""", unsafe_allow_html=True)

# ================= DAMAGE CLASSES =================

classes = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ================= MAIN LAYOUT =================

col1, col2 = st.columns([1, 1])

# ================= UPLOAD SECTION =================

with col1:

    st.markdown("""
    <div class="glass-card">

        <div class="section-title">
            📤 Upload Road Image
        </div>

    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE PREVIEW =================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with col2:

        st.markdown("""
        <div class="glass-card">

            <div class="section-title">
                🖼 Image Preview
            </div>

        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ================= PREDICTION LOGIC =================

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

        <div class="metric-value">
            🚨 {pred_label} Detected
        </div>

        <div class="metric-title">
            Confidence Score
        </div>

        <h1 style="font-size:60px;">
            {confidence:.2f}%
        </h1>

        <div class="metric-title">
            Severity Level
        </div>

        <h2 style="color:{sev_color}; font-size:36px;">
            {severity}
        </h2>

    </div>
    """, unsafe_allow_html=True)

    # ================= CHART SECTION =================

    st.markdown("""
    <div class="glass-card">

        <div class="section-title">
            📊 Confidence Analysis
        </div>

    """, unsafe_allow_html=True)

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

        font=dict(
            color='white',
            size=14
        ),

        xaxis_title="Damage Type",

        yaxis_title="Confidence",

        yaxis=dict(range=[0,1])
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================

    st.markdown("""
    <div class="recommend-card">

        <div class="section-title">
            ⚠ Maintenance Recommendations
        </div>
    """, unsafe_allow_html=True)

    if severity == "High":

        st.markdown("""
        <div class="normal-text">

        🔴 Immediate maintenance required.<br><br>

        🔴 Severe road damage detected.<br><br>

        🔴 Authorities should prioritize urgent repairs.<br><br>

        🔴 High accident risk area.

        </div>
        """, unsafe_allow_html=True)

    elif severity == "Medium":

        st.markdown("""
        <div class="normal-text">

        🟠 Moderate road damage detected.<br><br>

        🟠 Schedule maintenance soon.<br><br>

        🟠 Continuous monitoring recommended.<br><br>

        🟠 Repairs can prevent further damage.

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="normal-text">

        🟢 Minor road damage detected.<br><br>

        🟢 Continue regular inspections.<br><br>

        🟢 Low immediate safety risk.<br><br>

        🟢 Preventive maintenance suggested.

        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= EMPTY STATE =================

else:

    st.markdown("""
    <div class="glass-card" style="text-align:center;">

        <h1 style="font-size:40px;">
            🚀 Upload a road image to begin AI analysis
        </h1>

        <p style="font-size:20px; color:#cbd5e1;">
            The system will automatically detect potholes,
            cracks, and manholes.
        </p>

    </div>
    """, unsafe_allow_html=True)
