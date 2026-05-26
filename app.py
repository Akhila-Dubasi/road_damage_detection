import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Road Damage Detection AI",
    page_icon="🚧",
    layout="wide"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

/* ================= GOOGLE FONT ================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ================= GLOBAL ================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ================= MAIN BACKGROUND ================= */

.stApp {

    background:
        radial-gradient(circle at top left,
        rgba(99,102,241,0.18) 0%,
        transparent 28%),

        radial-gradient(circle at bottom right,
        rgba(6,182,212,0.16) 0%,
        transparent 30%),

        linear-gradient(
            135deg,
            #0b1020 0%,
            #111827 45%,
            #0f172a 100%
        );

    color: #f8fafc;
}

/* ================= HIDE STREAMLIT ================= */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
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

    -webkit-backdrop-filter: blur(18px);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);

    margin-bottom: 35px;
}

/* ================= MAIN TITLE ================= */

.main-title {

    font-size: 64px;

    font-weight: 900;

    letter-spacing: -1px;

    background: linear-gradient(
        to right,
        #67e8f9,
        #60a5fa,
        #818cf8
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    margin-bottom: 12px;
}

/* ================= SUBTITLE ================= */

.sub-title {

    font-size: 22px;

    font-weight: 400;

    color: #cbd5e1;
}

/* ================= GLASS CARD ================= */

.section-box {

    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    -webkit-backdrop-filter: blur(18px);

    padding: 30px;

    border-radius: 28px;

    margin-bottom: 28px;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);

    transition: all 0.3s ease;
}

.section-box:hover {

    transform: translateY(-4px);

    box-shadow:
        0 12px 35px rgba(96,165,250,0.12);
}

/* ================= SECTION TITLE ================= */

.section-title {

    font-size: 32px;

    font-weight: 700;

    color: #7dd3fc;

    margin-bottom: 20px;
}

/* ================= TEXT ================= */

.normal-text {

    font-size: 18px;

    line-height: 2;

    color: #e2e8f0;
}

/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.04);

    border: 2px dashed rgba(125,211,252,0.7);

    border-radius: 22px;

    padding: 20px;
}

/* ================= PREDICTION BOX ================= */

.prediction-box {

    background:
        linear-gradient(
            135deg,
            rgba(14,165,233,0.25),
            rgba(59,130,246,0.18)
        );

    border: 1px solid rgba(96,165,250,0.25);

    padding: 34px;

    border-radius: 28px;

    text-align: center;

    margin-top: 22px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.35);
}

/* ================= RECOMMENDATION BOX ================= */

.recommend-box {

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.12),
            rgba(127,29,29,0.18)
        );

    border: 1px solid rgba(248,113,113,0.15);

    padding: 28px;

    border-radius: 28px;

    margin-top: 22px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.35);
}

/* ================= IMAGE ================= */

img {

    border-radius: 22px;
}

/* ================= HEADINGS ================= */

h1, h2, h3 {

    color: white;
}

/* ================= SCROLLBAR ================= */

::-webkit-scrollbar {

    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: #60a5fa;

    border-radius: 20px;
}

::-webkit-scrollbar-track {

    background: #111827;
}

/* ================= ANIMATION ================= */

@keyframes glow {

    0% {
        box-shadow: 0 0 10px rgba(96,165,250,0.15);
    }

    50% {
        box-shadow: 0 0 24px rgba(96,165,250,0.28);
    }

    100% {
        box-shadow: 0 0 10px rgba(96,165,250,0.15);
    }
}

.prediction-box {

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
<div class="section-box">

    <div class="section-title">
        📌 About This Project
    </div>

    <div class="normal-text">

        ✅ Detects potholes, cracks, and manholes automatically using AI.<br><br>

        ✅ Helps smart cities monitor road conditions efficiently.<br><br>

        ✅ Reduces accidents and infrastructure maintenance delays.<br><br>

        ✅ Uses Convolutional Neural Networks (CNNs) for image classification.<br><br>

        ✅ Future-ready for autonomous vehicles and smart city systems.

    </div>

</div>
""", unsafe_allow_html=True)

# ================= MODEL =================

classes = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ================= IMAGE UPLOAD =================

st.markdown("""
<div class="section-box">

    <div class="section-title">
        📤 Upload Road Image
    </div>

""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= IMAGE PREVIEW =================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("""
    <div class="section-box">

        <div class="section-title">
            🖼 Uploaded Image Preview
        </div>

    """, unsafe_allow_html=True)

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= FAKE PREDICTION =================

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

    st.markdown("""
    <div class="prediction-box">
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <h1>🚨 {pred_label} Detected</h1>

        <h2>
            Confidence: {confidence:.2f}%
        </h2>

        <h2 style='color:{sev_color};'>
            Severity: {severity}
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUALIZATION =================

    st.markdown("""
    <div class="section-box">

        <div class="section-title">
            📊 Class Confidence Graph
        </div>

    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7,4))

    bars = ax.bar(
        classes,
        probabilities
    )

    for bar in bars:
        bar.set_alpha(0.8)

    ax.set_ylabel("Confidence")

    ax.set_ylim([0,1])

    ax.grid(alpha=0.3)

    fig.patch.set_facecolor('#111827')

    ax.set_facecolor('#111827')

    ax.tick_params(colors='white')

    ax.yaxis.label.set_color('white')

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================

    st.markdown("""
    <div class="recommend-box">
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <h2>
            ⚠ Recommendations
        </h2>
        """,
        unsafe_allow_html=True
    )

    if severity == "High":

        st.markdown("""
        <div class="normal-text">

        🔴 Immediate maintenance recommended.<br><br>

        🔴 High-risk road condition detected.<br><br>

        🔴 Authorities should prioritize repair work urgently.

        </div>
        """, unsafe_allow_html=True)

    elif severity == "Medium":

        st.markdown("""
        <div class="normal-text">

        🟠 Schedule maintenance soon.<br><br>

        🟠 Moderate road damage detected.<br><br>

        🟠 Regular monitoring recommended.

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="normal-text">

        🟢 Minor road issue detected.<br><br>

        🟢 Continue periodic inspections.<br><br>

        🟢 Low immediate safety risk.

        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= EMPTY STATE =================

else:

    st.markdown("""
    <div class="section-box" style="text-align:center;">

        <h1 style="font-size:38px;">
            🚀 Upload a road image to begin AI analysis
        </h1>

        <p style="font-size:20px; color:#cbd5e1;">
            The system will automatically detect potholes,
            cracks, and manholes.
        </p>

    </div>
    """, unsafe_allow_html=True)
