"""
Water Quality Classification — Streamlit Prediction App
Supports: Random Forest | XGBoost | Deep Learning (Keras)

NEW: For any feature you don't have a measurement for, the app asks
     3 proxy questions and estimates the value from your answers.

Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Feature meta ───────────────────────────────────────────────────────────────
FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

FEATURE_RANGES = {
    "ph":               (0.0,    14.0,    7.0),
    "Hardness":         (47.0,   323.0,   196.0),
    "Solids":           (320.0,  61227.0, 22014.0),
    "Chloramines":      (0.4,    13.1,    7.1),
    "Sulfate":          (129.0,  481.0,   333.0),
    "Conductivity":     (181.0,  753.0,   426.0),
    "Organic_carbon":   (2.2,    28.3,    14.3),
    "Trihalomethanes":  (0.7,    124.0,   66.4),
    "Turbidity":        (1.5,    6.9,     3.97),
}

CLASS_LABELS = {0: "❌ Not Potable", 1: "✅ Potable"}
CLASS_COLORS = {0: "#e74c3c",         1: "#2ecc71"}

# ── Model selector metadata ────────────────────────────────────────────────────
MODEL_META = {
    "Random Forest": {
        "icon":  "🌲",
        "tag":   "RF",
        "color": "#27ae60",
        "desc":  "Ensemble of decision trees. Robust, interpretable, great feature importance.",
    },
    "XGBoost": {
        "icon":  "⚡",
        "tag":   "XGB",
        "color": "#e67e22",
        "desc":  "Gradient-boosted trees. High accuracy, handles missing values natively.",
    },
    "Deep Learning": {
        "icon":  "🧠",
        "tag":   "DL",
        "color": "#8e44ad",
        "desc":  "Neural network. Learns complex non-linear patterns in the data.",
    },
    "Ensemble (All)": {
        "icon":  "🔮",
        "tag":   "ENS",
        "color": "#2980b9",
        "desc":  "Majority vote across all three models. Most reliable overall verdict.",
    },
}

# ── 20 water tips / fun facts ──────────────────────────────────────────────────
WATER_TIPS = [
    ("💧", "About 71% of the Earth's surface is covered by water — yet only 3% is freshwater."),
    ("🧊", "Most of Earth's freshwater (about 69%) is locked in glaciers and ice caps."),
    ("🧠", "The human brain is approximately 75% water. Mild dehydration can impair concentration."),
    ("⏱️", "A person can survive roughly a month without food, but only 3–5 days without water."),
    ("🌡️", "Hot water can freeze faster than cold water — a phenomenon called the Mpemba effect."),
    ("🚿", "A leaky faucet dripping once per second wastes over 3,000 gallons of water per year."),
    ("🐘", "An elephant can smell water from up to 3 miles away."),
    ("🌊", "Water is the only natural substance found in all three states — liquid, solid, and gas — at temperatures normal on Earth."),
    ("🧪", "pH 7 is neutral. WHO guidelines recommend drinking water between pH 6.5 and 8.5."),
    ("🔬", "Turbidity measures water cloudiness. High turbidity can signal bacterial or chemical contamination."),
    ("🏭", "Chlorine has been used to disinfect public drinking water since the early 1900s, dramatically reducing waterborne diseases."),
    ("⚗️", "Boiling water kills most biological pathogens but does NOT remove dissolved chemicals or heavy metals."),
    ("🪨", "Water hardness is caused by dissolved calcium and magnesium — it's not a health risk but affects appliances and soap."),
    ("🌱", "Producing 1 kg of beef requires approximately 15,000 litres of water across the full supply chain."),
    ("🌍", "About 2.2 billion people worldwide lack access to safely managed drinking water services (WHO, 2023)."),
    ("💦", "The average adult needs about 2–3 litres of water per day, depending on activity level and climate."),
    ("🏔️", "Groundwater stored in aquifers accounts for about 30% of all freshwater on Earth."),
    ("🫧", "Total Dissolved Solids (TDS) measures all minerals, salts, and metals dissolved in water — safe levels are under 500 mg/L."),
    ("🌀", "A water molecule (H₂O) can travel from ocean to cloud to rain and back to the ocean in as little as 9 days."),
    ("🧫", "Drinking water can contain over 300 different chemical compounds — regular quality testing is essential."),
]

# ══════════════════════════════════════════════════════════════════════════════
# PROXY QUESTION BANK
# Each feature has 3 questions.  Every option maps to a score in [0.0, 1.0]
# representing where in [lo, hi] the estimated value should sit.
# Final estimate = lo + mean(scores) * (hi - lo)
# ══════════════════════════════════════════════════════════════════════════════

PROXY_QUESTIONS = {
    "ph": [
        {
            "q": "What do metal pipes, faucets, or containers exposed to this water look like?",
            "options": {
                "Blue-green staining — clear corrosion of copper/metal":  0.10,
                "Reddish-brown rust stains on iron/steel surfaces":        0.30,
                "Dull or slightly discoloured surfaces":                   0.50,
                "White chalky/scaly deposits on surfaces":                 0.75,
                "Surfaces look normal — no visible corrosion or scaling":  0.50,
            },
        },
        {
            "q": "Does soap lather easily in this water?",
            "options": {
                "Very easily — produces abundant foam":          0.75,
                "Normally — average foam":                       0.50,
                "Poorly — takes a lot of soap to lather":        0.25,
            },
        },
        {
            "q": "What is the primary source of this water?",
            "options": {
                "Municipal treated supply":                      0.55,
                "Limestone or mineral-rich groundwater":         0.70,
                "Mountain spring or rainfall":                   0.40,
                "Industrial or mining area":                     0.25,
            },
        },
    ],

    "Hardness": [
        {
            "q": "Do you notice limescale or white deposits on faucets/kettles?",
            "options": {
                "Heavy crust builds up quickly":                 0.90,
                "Some deposits after a while":                   0.60,
                "Occasional light film":                         0.35,
                "None at all":                                   0.10,
            },
        },
        {
            "q": "How does soap or shampoo behave in this water?",
            "options": {
                "Hard to rinse off — leaves a sticky feeling":   0.85,
                "Rinses normally":                               0.50,
                "Rinses extremely easily — water feels silky":   0.15,
            },
        },
        {
            "q": "How does laundry feel after washing with this water?",
            "options": {
                "Stiff or rough — fabrics feel scratchy":        0.80,
                "Normal feel":                                   0.50,
                "Unusually soft":                                0.20,
            },
        },
    ],

    "Solids": [
        {
            "q": "Does the water leave a white film or spots on glass, sinks, or dishes when it dries?",
            "options": {
                "Heavy white spots and film on everything":       0.90,
                "Noticeable spots on glass and dark surfaces":    0.65,
                "Faint spotting on glass only":                   0.40,
                "No spots or film at all":                        0.10,
            },
        },
        {
            "q": "What residue is left after boiling water in a pot or kettle?",
            "options": {
                "Heavy white/grey crust":                        0.90,
                "Moderate white film":                           0.60,
                "Light film":                                    0.35,
                "No visible residue":                            0.10,
            },
        },
        {
            "q": "What is the primary source of this water?",
            "options": {
                "Deep groundwater or artesian well":             0.80,
                "Shallow well or borehole":                      0.55,
                "River, lake, or reservoir":                     0.35,
                "Rainwater or distilled":                        0.05,
            },
        },
    ],

    "Chloramines": [
        {
            "q": "Does the water smell of bleach or chlorine?",
            "options": {
                "Strong bleach smell":                           0.90,
                "Noticeable chemical smell":                     0.65,
                "Very faint chemical hint":                      0.35,
                "No chemical smell at all":                      0.05,
            },
        },
        {
            "q": "Is your water supply municipally or publicly treated?",
            "options": {
                "Yes — heavily treated city supply":             0.85,
                "Yes — standard municipal treatment":            0.55,
                "Partially treated (small utility)":             0.30,
                "No — private well or untreated source":         0.05,
            },
        },
        {
            "q": "Do you notice skin dryness, irritation, or eye redness after bathing/washing with this water?",
            "options": {
                "Strong irritation — skin very dry, eyes red":   0.85,
                "Mild dryness or slight eye irritation":         0.55,
                "Very slight or occasional irritation":          0.30,
                "No irritation at all":                          0.05,
            },
        },
    ],

    "Sulfate": [
        {
            "q": "Does the water have a 'rotten egg' or sulphur smell?",
            "options": {
                "Strong rotten-egg odour":                       0.92,
                "Mild sulphur odour":                            0.65,
                "Very faint odour":                              0.35,
                "No odour":                                      0.10,
            },
        },
        {
            "q": "Is your water source near industrial, mining, or agricultural activity?",
            "options": {
                "Yes — heavy industrial or mining nearby":       0.85,
                "Yes — farming / agricultural area":             0.60,
                "Suburban — some industry in region":            0.40,
                "No — rural or protected watershed":             0.15,
            },
        },
        {
            "q": "Have you or others noticed digestive discomfort (laxative effect) from this water?",
            "options": {
                "Yes — frequently":                              0.88,
                "Occasionally — mostly for newcomers":           0.60,
                "Rarely":                                        0.35,
                "Never":                                         0.15,
            },
        },
    ],

    "Conductivity": [
        {
            "q": "How would you describe the overall mineral 'feel' of the water?",
            "options": {
                "Very heavy — strongly mineral":                 0.90,
                "Moderately mineral":                            0.60,
                "Lightly mineral":                               0.35,
                "Completely pure / no mineral character":        0.10,
            },
        },
        {
            "q": "What type of geographical area does the water come from?",
            "options": {
                "Industrial or heavily urbanised area":          0.85,
                "Suburban / mixed use":                          0.60,
                "Rural farmland":                                0.45,
                "Mountainous or forested watershed":             0.20,
            },
        },
        {
            "q": "Does your water system include a softener or demineraliser?",
            "options": {
                "No treatment at all":                           0.80,
                "Basic filtration only":                         0.60,
                "Partial softening":                             0.40,
                "Full softening / reverse osmosis":              0.10,
            },
        },
    ],

    "Organic_carbon": [
        {
            "q": "What colour does the water appear?",
            "options": {
                "Distinctly yellow or brownish":                 0.90,
                "Very slightly tinted":                          0.60,
                "Completely clear":                              0.20,
            },
        },
        {
            "q": "Is the water source near agricultural land, wetlands, or dense forest?",
            "options": {
                "Yes — directly adjacent to farmland or swamp":  0.85,
                "Nearby — within a few kilometres":              0.60,
                "Somewhat — general rural area":                 0.40,
                "No — urban or well-protected watershed":        0.15,
            },
        },
        {
            "q": "Does the water have an earthy, musty, or 'soil-like' smell?",
            "options": {
                "Strong earthy/musty odour":                     0.88,
                "Noticeable earthy hint":                        0.60,
                "Very faint":                                    0.35,
                "No odour":                                      0.10,
            },
        },
    ],

    "Trihalomethanes": [
        {
            "q": "Is the water treated with chlorine or chloramines?",
            "options": {
                "Yes — heavily chlorinated municipal supply":     0.90,
                "Yes — standard chlorination":                   0.60,
                "Minimal disinfection":                          0.25,
                "Not chlorinated (private well / spring)":       0.05,
            },
        },
        {
            "q": "Is the water source surface water (river, lake, reservoir)?",
            "options": {
                "Yes — main source is surface water":            0.80,
                "Mixed — surface and groundwater blend":         0.55,
                "Primarily groundwater":                         0.30,
                "100 % groundwater / well":                      0.10,
            },
        },
        {
            "q": "How far does the water travel through pipes before reaching you?",
            "options": {
                "Long distribution network (large city)":        0.85,
                "Medium distribution (town)":                    0.55,
                "Short distance (small community)":              0.30,
                "Very direct / on-site source":                  0.08,
            },
        },
    ],

    "Turbidity": [
        {
            "q": "How does the water look when held up to light in a clear glass?",
            "options": {
                "Cloudy or murky — can't see through easily":    0.90,
                "Slightly hazy":                                 0.60,
                "Almost clear with a faint haze":                0.35,
                "Perfectly clear and transparent":               0.10,
            },
        },
        {
            "q": "Can you see visible particles or sediment in the water?",
            "options": {
                "Yes — many visible particles":                  0.95,
                "Yes — occasional floating bits":                0.65,
                "None visible, but water feels gritty":          0.40,
                "Completely particle-free":                      0.08,
            },
        },
        {
            "q": "After heavy rainfall, how much does the water quality change?",
            "options": {
                "Becomes very cloudy / unusable":                0.90,
                "Noticeably cloudier":                           0.65,
                "Slightly worse":                                0.35,
                "No change at all":                              0.10,
            },
        },
    ],
}


def estimate_from_proxy(feature: str, answers: list[float]) -> float:
    """Convert three proxy scores [0,1] → actual feature value."""
    lo, hi, _ = FEATURE_RANGES[feature]
    return round(lo + np.mean(answers) * (hi - lo), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Mock model helpers
# ══════════════════════════════════════════════════════════════════════════════

class MockClassifier:
    def __init__(self, name): self.name = name
    def predict(self, X): return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    def predict_proba(self, X):
        score = np.clip(np.mean(X, axis=1) / 10 + 0.45, 0.05, 0.95)
        return np.column_stack([1 - score, score])
    @property
    def feature_importances_(self):
        rng = np.random.RandomState(0)
        raw = rng.rand(len(FEATURES))
        return raw / raw.sum()

class MockKerasModel:
    def predict(self, X):
        score = np.clip(np.mean(X, axis=1) / 10 + 0.45, 0.05, 0.95)
        return score.reshape(-1, 1)

@st.cache_resource
def load_models():
    MODEL_DIR  = "deployed_models"
    rf_path     = os.path.join(MODEL_DIR, "rf_model.pkl")
    xgb_path    = os.path.join(MODEL_DIR, "xgb_model.pkl")
    dl_path     = os.path.join(MODEL_DIR, "dl_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "robust_scaler.pkl")

    rf  = joblib.load(rf_path)  if os.path.exists(rf_path)  else MockClassifier("Random Forest")
    xgb = joblib.load(xgb_path) if os.path.exists(xgb_path) else MockClassifier("XGBoost")

    if KERAS_AVAILABLE and os.path.exists(dl_path):
        dl = keras.models.load_model(dl_path)
    else:
        dl = MockKerasModel()

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Sidebar status indicator
    real = os.path.exists(rf_path)
    if real:
        st.sidebar.success("✅ Real models loaded", icon="🎯")
    else:
        st.sidebar.warning("⚠️ Using mock models", icon="🔧")

    return rf, xgb, dl, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_with(models, X: np.ndarray, selected: str) -> dict:
    """Scale input then run prediction for the selected model or all (Ensemble)."""
    rf, xgb, dl, scaler = models

    # Apply RobustScaler if available (matches training pipeline)
    X_scaled = scaler.transform(X) if scaler is not None else X

    results = {}
    runners = {
        "Random Forest": lambda: rf.predict_proba(X_scaled)[:, 1],
        "XGBoost":       lambda: xgb.predict_proba(X_scaled)[:, 1],
        "Deep Learning": lambda: dl.predict(X_scaled).flatten(),
    }
    targets = list(runners.keys()) if selected == "Ensemble (All)" else [selected]
    for name in targets:
        prob = runners[name]()
        results[name] = {"prob": prob, "label": (prob > 0.5).astype(int)}
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ══════════════════════════════════════════════════════════════════════════════

def gauge_chart(prob, title):
    fig, ax = plt.subplots(figsize=(3.2, 1.8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    color = CLASS_COLORS[int(prob > 0.5)]
    ax.barh(0, prob,       color=color,     height=0.4)
    ax.barh(0, 1 - prob,   color="#2d2d2d", height=0.4, left=prob)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.axis("off")
    ax.text(0.5, -0.35, f"{prob:.1%}", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.set_title(title, color="white", fontsize=10, pad=4)
    plt.tight_layout()
    return fig

def confidence_comparison_chart(results):
    names  = list(results.keys())
    probs  = [results[n]["prob"][0] for n in names]
    colors = [CLASS_COLORS[int(p > 0.5)] for p in probs]
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#0e1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(names, probs, color=colors, width=0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Potability Probability", color="white")
    ax.set_title("Model Confidence Comparison", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.02,
                f"{p:.1%}", ha="center", color="white", fontsize=9)
    plt.tight_layout()
    return fig

def feature_importance_chart(model, title):
    imp = model.feature_importances_
    if callable(imp): imp = imp()
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")
    ax.barh([FEATURES[i] for i in idx], imp[idx], color="#3498db")
    ax.set_xlabel("Importance", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Feature input widget — direct slider OR 3 proxy questions
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_UNITS = {
    "ph":              "pH units",
    "Hardness":        "mg/L",
    "Solids":          "ppm",
    "Chloramines":     "ppm",
    "Sulfate":         "mg/L",
    "Conductivity":    "μS/cm",
    "Organic_carbon":  "ppm",
    "Trihalomethanes": "μg/L",
    "Turbidity":       "NTU",
}

def feature_input_widget(feature: str) -> tuple[float, bool]:
    """
    Returns (value, was_estimated).
    Shows slider if user has measurement; otherwise shows 3 proxy questions.
    """
    lo, hi, default = FEATURE_RANGES[feature]
    unit = FEATURE_UNITS.get(feature, "")

    col_label, col_toggle = st.columns([3, 1])
    with col_label:
        st.markdown(f"**{feature}** `{unit}`")
    with col_toggle:
        no_data = st.toggle("Estimate →", key=f"toggle_{feature}",
                            help="I don't have this measurement — ask me questions instead")

    if not no_data:
        val = st.slider(
            f"{feature} value",
            min_value=float(lo), max_value=float(hi), value=float(default),
            key=f"slider_{feature}", label_visibility="collapsed",
        )
        return val, False

    # ── Proxy question path ────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#1a2233;padding:10px 14px;border-radius:8px;"
        f"border-left:3px solid #3498db;margin-bottom:6px'>"
        f"<small>Answer these 3 questions to estimate <b>{feature}</b>:</small></div>",
        unsafe_allow_html=True,
    )
    scores = []
    for i, q_data in enumerate(PROXY_QUESTIONS[feature]):
        opts = list(q_data["options"].keys())
        choice = st.radio(
            f"Q{i+1}: {q_data['q']}",
            options=opts,
            key=f"proxy_{feature}_{i}",
            index=len(opts)//2,          # default to middle option
        )
        scores.append(q_data["options"][choice])

    estimated = estimate_from_proxy(feature, scores)
    st.info(f"📐 Estimated **{feature}**: `{estimated}` {unit}  "
            f"*(based on your answers)*", icon="💡")
    return estimated, True


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 💧 Water Quality Predictor")
        st.markdown("---")
        mode = st.radio("Mode", ["🔬 Single Prediction", "📂 Batch CSV Prediction"])
        st.markdown("---")
        st.info(
            "**Tip:** Toggle **Estimate →** next to any feature you don't have "
            "a measurement for. The app will ask 3 questions to estimate it.",
            icon="💡",
        )
        st.caption("Swap mock models with your real .joblib / .h5 files.")
    return mode


# ══════════════════════════════════════════════════════════════════════════════
# Model selector widget
# ══════════════════════════════════════════════════════════════════════════════

def render_model_selector() -> str:
    """Card-style model selector. Returns the name of the selected model."""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Ensemble (All)"

    st.markdown("### 🤖 Choose a Model")
    cols = st.columns(4)
    for col, (name, meta) in zip(cols, MODEL_META.items()):
        with col:
            is_active = st.session_state.selected_model == name
            border_color = meta["color"] if is_active else "#2d2d2d"
            bg_color     = meta["color"] + "22" if is_active else "#161b22"
            st.markdown(
                f"""<div style='border:2px solid {border_color};border-radius:10px;
                    padding:12px 8px;background:{bg_color};text-align:center;
                    min-height:110px;'>
                  <div style='font-size:26px'>{meta["icon"]}</div>
                  <div style='font-weight:700;color:white;font-size:13px;margin:4px 0'>{name}</div>
                  <div style='font-size:9px;color:#aaa;line-height:1.3'>{meta["desc"]}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            # invisible button that covers the card area
            if st.button(f"Select {meta['tag']}", key=f"mdl_{name}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.selected_model = name
                st.rerun()

    selected = st.session_state.selected_model
    m = MODEL_META[selected]
    st.markdown(
        f"<div style='margin-top:6px;padding:6px 12px;border-radius:6px;"
        f"background:{m['color']}22;border-left:3px solid {m['color']}'>"
        f"<b style='color:{m['color']}'>{m['icon']} {selected}</b> — {m['desc']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Water tips carousel
# ══════════════════════════════════════════════════════════════════════════════

def render_water_tips():
    tips_js = str([(e, t) for e, t in WATER_TIPS]).replace("'", "\'")
    html = f"""
    <style>
      #tip-wrap {{
        margin: 14px 0 4px 0;
        padding: 14px 18px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
        border: 1px solid #1e3a5f;
        min-height: 70px;
        display: flex;
        align-items: center;
        gap: 14px;
      }}
      #tip-emoji {{ font-size: 28px; flex-shrink: 0; }}
      #tip-label {{
        font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
        color: #3498db; text-transform: uppercase; margin-bottom: 3px;
      }}
      #tip-text {{
        color: #dce8f5; font-size: 13px; line-height: 1.5;
        transition: opacity 0.5s ease;
      }}
    </style>
    <div id="tip-wrap">
      <div id="tip-emoji">💧</div>
      <div>
        <div id="tip-label">💡 Did you know?</div>
        <div id="tip-text">Loading tip...</div>
      </div>
    </div>
    <script>
      const tips = {tips_js};
      let idx = Math.floor(Math.random() * tips.length);

      function showTip() {{
        const el = document.getElementById('tip-text');
        const em = document.getElementById('tip-emoji');
        el.style.opacity = 0;
        setTimeout(() => {{
          el.textContent = tips[idx][1];
          em.textContent = tips[idx][0];
          el.style.opacity = 1;
          idx = (idx + 1) % tips.length;
        }}, 500);
      }}

      showTip();
      setInterval(showTip, 6000);
    </script>
    """
    st.components.v1.html(html, height=100)


# ══════════════════════════════════════════════════════════════════════════════
# Single prediction UI
# ══════════════════════════════════════════════════════════════════════════════

def single_prediction_ui(models):
    # ── Model selector ─────────────────────────────────────────────────────
    selected_model = render_model_selector()

    st.subheader("🔬 Single Sample Prediction")
    st.markdown(
        "Enter values you have directly, or **toggle 'Estimate →'** on any "
        "feature you don't have — the app will ask 3 questions to estimate it."
    )
    st.markdown("---")

    input_vals      = {}
    estimated_flags = {}

    pairs = [FEATURES[i:i+2] for i in range(0, len(FEATURES), 2)]
    for pair in pairs:
        cols = st.columns(len(pair))
        for col, feat in zip(cols, pair):
            with col:
                with st.container(border=True):
                    val, was_estimated = feature_input_widget(feat)
                    input_vals[feat]      = val
                    estimated_flags[feat] = was_estimated
        st.markdown("")

    estimated_feats = [f for f, est in estimated_flags.items() if est]
    if estimated_feats:
        st.caption(f"🔮 Estimated features: {', '.join(estimated_feats)}")

    st.markdown("---")

    predict_clicked = st.button("⚡ Predict", use_container_width=True, type="primary")

    # ── Tips carousel (always visible below predict button) ─────────────────
    render_water_tips()

    if predict_clicked:
        X       = np.array([[input_vals[f] for f in FEATURES]])
        results = predict_with(models, X, selected_model)
        meta    = MODEL_META[selected_model]

        st.markdown("---")
        st.subheader("Prediction Results")

        # Model badge
        st.markdown(
            f"<div style='display:inline-block;padding:4px 12px;border-radius:20px;"
            f"background:{meta['color']}33;border:1px solid {meta['color']};"
            f"color:{meta['color']};font-weight:700;font-size:13px;margin-bottom:12px'>"
            f"{meta['icon']} {selected_model}</div>",
            unsafe_allow_html=True,
        )

        if selected_model == "Ensemble (All)":
            # Show all three gauges + comparison chart
            gcols = st.columns(3)
            for col, (name, res) in zip(gcols, results.items()):
                with col:
                    st.markdown(f"**{name}**")
                    st.pyplot(gauge_chart(res["prob"][0], CLASS_LABELS[res["label"][0]]),
                              use_container_width=True)
            st.pyplot(confidence_comparison_chart(results), use_container_width=True)
            votes    = [res["label"][0] for res in results.values()]
            majority = int(np.round(np.mean(votes)))
            verdict_label = CLASS_LABELS[majority]
            verdict_color = CLASS_COLORS[majority]
            verdict_prefix = "Majority Verdict"
        else:
            # Single model — one big gauge
            res    = results[selected_model]
            prob   = res["prob"][0]
            label  = res["label"][0]
            gcol, _ = st.columns([1, 2])
            with gcol:
                st.pyplot(gauge_chart(prob, CLASS_LABELS[label]), use_container_width=True)
            majority      = label
            verdict_label = CLASS_LABELS[label]
            verdict_color = CLASS_COLORS[label]
            verdict_prefix = f"{meta['icon']} {selected_model} Verdict"

        st.markdown(
            f"<div style='padding:14px;border-radius:8px;"
            f"background:{verdict_color}22;border:1px solid {verdict_color};"
            f"text-align:center;font-size:18px;font-weight:700;color:{verdict_color}'>"
            f"{verdict_prefix}: {verdict_label}</div>",
            unsafe_allow_html=True,
        )

        # Feature importance — only for tree models
        tree_models = {"Random Forest": models[0], "XGBoost": models[1]}
        show_fi = (
            selected_model in tree_models or selected_model == "Ensemble (All)"
        )
        if show_fi:
            st.markdown("---")
            st.subheader("Feature Importance")
            fi_targets = (
                list(tree_models.items())
                if selected_model == "Ensemble (All)"
                else [(selected_model, tree_models[selected_model])]
            )
            fi_cols = st.columns(len(fi_targets))
            for col, (name, mdl) in zip(fi_cols, fi_targets):
                with col:
                    st.pyplot(feature_importance_chart(mdl, name), use_container_width=True)

        # Input summary table
        st.markdown("---")
        st.subheader("Input Summary")
        summary = pd.DataFrame({
            "Feature": FEATURES,
            "Value":   [round(input_vals[f], 4) for f in FEATURES],
            "Source":  ["🔮 Estimated" if estimated_flags[f] else "📏 Direct" for f in FEATURES],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Batch prediction UI
# ══════════════════════════════════════════════════════════════════════════════

def batch_prediction_ui(models):
    st.subheader("📂 Batch Prediction from CSV")
    st.markdown(
        f"Upload a CSV with any/all of these columns: **{', '.join(FEATURES)}**. "
        "Missing columns will be filled with the dataset median."
    )

    sample_df = pd.DataFrame({f: [FEATURE_RANGES[f][2]] for f in FEATURES})
    st.download_button("⬇️ Download sample CSV template",
                       data=sample_df.to_csv(index=False),
                       file_name="water_sample_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            # Fill missing feature columns with the dataset median (default)
            for feat in FEATURES:
                if feat not in df.columns:
                    df[feat] = FEATURE_RANGES[feat][2]
                    st.warning(f"Column **{feat}** not found — filled with median `{FEATURE_RANGES[feat][2]}`")

            X = df[FEATURES].values
            results = predict_with(models, X, 'Ensemble (All)')

            for name, res in results.items():
                safe = name.replace(" ", "_")
                df[f"{safe}_prediction"] = [CLASS_LABELS[l] for l in res["label"]]
                df[f"{safe}_confidence"] = (res["prob"] * 100).round(2)

            st.success(f"✅ Predictions complete for {len(df)} samples.")
            st.dataframe(df, use_container_width=True)

            st.download_button("⬇️ Download predictions CSV",
                               data=df.to_csv(index=False).encode(),
                               file_name="predictions.csv",
                               mime="text/csv", type="primary")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mode   = render_sidebar()
    models = load_models()

    st.title("💧 Water Quality Classification")
    st.markdown(
        "Predict water potability using **Random Forest**, **XGBoost**, and "
        "**Deep Learning** — with ensemble majority voting."
    )
    st.markdown("---")

    if "Single" in mode:
        single_prediction_ui(models)
    else:
        batch_prediction_ui(models)

if __name__ == "__main__":
    main()
