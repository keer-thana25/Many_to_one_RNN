import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Custom CSS — cyber-security / messaging theme
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(20,184,166,0.16) 0%, transparent 40%),
            radial-gradient(circle at 92% 15%, rgba(239,68,68,0.12) 0%, transparent 40%),
            radial-gradient(circle at 50% 100%, rgba(59,130,246,0.14) 0%, transparent 50%),
            linear-gradient(160deg, #0a1418 0%, #0d1620 45%, #0a0e16 100%);
        color: #e6f4f1;
    }

    #MainMenu, footer {visibility: hidden;}

    /* Hero */
    .hero-wrap {
        text-align: center;
        padding: 2.2rem 1.5rem 1.7rem 1.5rem;
        margin-bottom: 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(20,184,166,0.12), rgba(59,130,246,0.1));
        border: 1px solid rgba(94,234,212,0.18);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        position: relative;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #5eead4, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #9db4c0;
        margin-top: 0.6rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,213,204,0.12);
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.3rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #eaf6f4;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-caption {
        color: #8fa3ae;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    /* Text area: force legible styling regardless of system theme */
    div[data-testid="stTextArea"] label p,
    div[data-testid="stTextArea"] label span {
        color: #9fd8cf !important;
        font-weight: 600 !important;
    }
    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        color: #e6f4f1 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(94,234,212,0.2) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid rgba(94,234,212,0.55) !important;
        box-shadow: 0 0 0 2px rgba(94,234,212,0.15) !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        height: 3.1rem;
        width: 100%;
        font-weight: 700;
        font-size: 1.02rem;
        border: none;
        background: linear-gradient(90deg, #0d9488, #2563eb);
        color: white;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(13,148,136,0.35);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 24px rgba(37,99,235,0.45);
    }

    /* Result banners */
    .result-spam {
        background: linear-gradient(120deg, rgba(239,68,68,0.16), rgba(239,68,68,0.06));
        border: 1px solid rgba(239,68,68,0.4);
        border-left: 6px solid #ef4444;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-top: 0.5rem;
        box-shadow: 0 6px 20px rgba(239,68,68,0.12);
    }
    .result-ham {
        background: linear-gradient(120deg, rgba(52,211,153,0.16), rgba(52,211,153,0.06));
        border: 1px solid rgba(52,211,153,0.4);
        border-left: 6px solid #34d399;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-top: 0.5rem;
        box-shadow: 0 6px 20px rgba(52,211,153,0.12);
    }
    .result-title-spam {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fca5a5;
        margin: 0 0 0.3rem 0;
    }
    .result-title-ham {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6ee7b7;
        margin: 0 0 0.3rem 0;
    }
    .result-caption {
        color: #b8c9c4;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    /* Confidence meter */
    .meter-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .meter-fill-spam {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #f87171, #ef4444);
    }
    .meter-fill-ham {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #34d399, #10b981);
    }
    .meter-label {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #9db4c0;
        margin-top: 0.4rem;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,213,204,0.12);
        border-radius: 12px;
        padding: 0.7rem 0.5rem;
    }
    [data-testid="stMetricLabel"] { color: #8fa3ae !important; }
    [data-testid="stMetricValue"] { color: #eaf6f4 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b1f 0%, #0a0e16 100%);
        border-right: 1px solid rgba(94,234,212,0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #5eead4;
        font-weight: 700;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-teal  { background: rgba(20,184,166,0.15); color: #5eead4; border: 1px solid rgba(20,184,166,0.35); }
    .badge-blue  { background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.35); }
    .badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.35); }

    hr { border-color: rgba(148,213,204,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Hero Header
# =====================================================================
st.markdown("""
<div class="hero-wrap">
    <p class="hero-title">🛡️ SMS Spam Detection</p>
    <p class="hero-subtitle">A Many-to-One RNN that reads a message and decides: Spam 🚨 or Ham ✅</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# Load model and tokenizer
# =====================================================================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('model/spam_model.keras')
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
    maxlen = model.input_shape[1]
    load_error = None
except Exception as e:
    load_error = str(e)

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("# 🛡️ Model Info")
    st.markdown("---")
    if load_error is None:
        st.markdown('<span class="badge badge-teal">✅ Model Loaded</span>', unsafe_allow_html=True)
        st.info(f"**Architecture**  \nMany-to-One RNN")
        st.info(f"**Max Sequence Length**  \n`{maxlen}` tokens")
        st.info(f"**Vocabulary Size**  \n`{len(tokenizer.word_index):,}` words")
    else:
        st.markdown('<span class="badge badge-amber">⚠️ Model Not Loaded</span>', unsafe_allow_html=True)
        st.error(load_error)

    st.markdown("---")
    st.markdown("### 🧠 How it works")
    st.caption(
        "The message is tokenized, padded to the model's expected length, "
        "then passed through the trained RNN which outputs a spam probability "
        "between 0 (Ham) and 1 (Spam)."
    )
    st.markdown("---")
    st.caption("Built with 🛡️ using TensorFlow + Streamlit")

if load_error is not None:
    st.stop()

# =====================================================================
# Input Section
# =====================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">✉️ Message Scanner</p>', unsafe_allow_html=True)
st.markdown('<p class="section-caption">Paste or type an SMS message below and scan it for spam.</p>', unsafe_allow_html=True)

user_input = st.text_area(
    "Message to check:",
    "Congratulations! You've won a $1000 gift card. Reply WIN to claim now.",
    height=140
)

check_btn = st.button("🔍 Scan Message", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# Result Section
# =====================================================================
if check_btn:
    if user_input.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        with st.spinner("🔎 Analyzing message..."):
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=maxlen)
            pred = model.predict(padded)[0][0]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Scan Result</p>', unsafe_allow_html=True)

        if pred > 0.5:
            confidence = pred * 100
            st.markdown(f"""
            <div class="result-spam">
                <p class="result-title-spam">🚨 Spam Detected</p>
                <p class="result-caption">This message shows strong signs of being spam.</p>
                <div class="meter-track"><div class="meter-fill-spam" style="width:{confidence:.1f}%;"></div></div>
                <div class="meter-label"><span>Spam Confidence</span><span>{confidence:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = (1 - pred) * 100
            st.markdown(f"""
            <div class="result-ham">
                <p class="result-title-ham">✅ Safe (Ham)</p>
                <p class="result-caption">This message looks like a legitimate, non-spam text.</p>
                <div class="meter-track"><div class="meter-fill-ham" style="width:{confidence:.1f}%;"></div></div>
                <div class="meter-label"><span>Ham Confidence</span><span>{confidence:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Raw Spam Probability", f"{pred:.1%}")
        m2.metric("Message Length", f"{len(user_input)} chars")

        st.markdown('</div>', unsafe_allow_html=True)
