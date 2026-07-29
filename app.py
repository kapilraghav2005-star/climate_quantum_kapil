# python -m streamlit run app.py

"""
Quantum AQI Predictor — Streamlit Frontend
============================================
A futuristic client for a FastAPI backend serving a Hybrid Quantum-Classical
Neural Network (PyTorch + PennyLane, 7-qubit variational circuit) that
predicts Air Quality Index (AQI).

Features:
  - Cyberpunk/dark glassmorphism UI
  - Fully gate-level decomposed circuit diagram (qml.draw_mpl, level="device")
  - Interactive 3D Plotly Bloch sphere
  - Automatic spoken AQI announcement via the browser's Web Speech API
  - Robust error handling for a locally-hosted FastAPI backend

Run with:  streamlit run app.py
Requires:  streamlit, requests, pennylane, matplotlib, plotly, numpy
Voice feedback uses the browser's built-in Web Speech API — no extra
Python TTS dependency (e.g. gTTS) is required, and it works fully offline.
"""

import time
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import time

# PennyLane is optional at runtime — the circuit diagram is a *visual mock*
# of the backend's real 7-qubit circuit, not a live inference call.
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

# API key is now securely fetched from Streamlit Secrets
API_KEY = st.secrets["OPENWEATHER_API_KEY"]
def get_live_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            st.sidebar.error(f"Weather API Error: {data.get('message', 'Unknown Error')}")
            return None
    except Exception as e:
        st.sidebar.error(f"Failed to fetch weather: {e}")
        return None

# ----------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

#API_URL = "http://127.0.0.1:8000/predict_aqi"
API_URL = "https://qai-climate-analysis.onrender.com/predict_aqi"

N_QUBITS = 7

# ----------------------------------------------------------------------------
# CUSTOM CSS — dark, futuristic, "hackathon-winning" aesthetic
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Rajdhani', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 20%, #0d1b2a 0%, #060912 55%, #020308 100%);
            color: #e6f1ff;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0f1e 0%, #0d1526 100%);
            border-right: 1px solid rgba(0, 255, 255, 0.15);
        }

        /* Headline / title styling */
        .quantum-title {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            font-size: 2.6rem;
            background: linear-gradient(90deg, #00f5ff, #7dffea, #00f5ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 245, 255, 0.25);
            letter-spacing: 1px;
            margin-bottom: 0;
        }
        .quantum-subtitle {
            font-family: 'Rajdhani', sans-serif;
            color: #7f9bbf;
            font-size: 1.05rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: -6px;
        }

        /* Glassmorphism cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 245, 255, 0.15);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(6px);
            margin-bottom: 1rem;
        }

        .section-header {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.05rem;
            color: #00f5ff;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(0, 245, 255, 0.2);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }

        /* Buttons */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #00c2ff, #00ffb3);
            color: #001018;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            letter-spacing: 1px;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 0;
            transition: all 0.25s ease;
            box-shadow: 0 0 20px rgba(0, 255, 200, 0.25);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 255, 200, 0.5);
            color: #001018;
        }

        /* Metric-like AQI display */
        .aqi-display {
            border-radius: 22px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 0 40px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .aqi-number {
            font-family: 'Orbitron', sans-serif;
            font-size: 4.2rem;
            font-weight: 900;
            line-height: 1;
            margin: 0;
        }
        .aqi-label {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.3rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            opacity: 0.9;
        }
        .aqi-category {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            letter-spacing: 2px;
            margin-top: 0.4rem;
        }

        hr {
            border-color: rgba(0, 245, 255, 0.15);
        }

        .footer-note {
            text-align: center;
            color: #4d6a8a;
            font-size: 0.8rem;
            letter-spacing: 1px;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def get_aqi_category(aqi: float):
    """Map a numeric AQI value to (label, color, glow-color) per standard bands."""
    if aqi <= 50:
        return "Good", "#00e676", "rgba(0, 230, 118, 0.25)"
    elif aqi <= 100:
        return "Moderate", "#ffee58", "rgba(255, 238, 88, 0.25)"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "#ffa726", "rgba(255, 167, 38, 0.25)"
    elif aqi <= 200:
        return "Unhealthy", "#ef5350", "rgba(239, 83, 80, 0.3)"
    elif aqi <= 300:
        return "Very Unhealthy", "#ab47bc", "rgba(171, 71, 188, 0.3)"
    else:
        return "Hazardous", "#8d0000", "rgba(141, 0, 0, 0.35)"


def render_aqi_display(aqi: float):
    """Render a large, color-coded AQI readout as custom HTML."""
    label, color, glow = get_aqi_category(aqi)
    st.markdown(
        f"""
        <div class="aqi-display" style="background: {glow};">
            <div class="aqi-label" style="color:{color};">Predicted AQI</div>
            <p class="aqi-number" style="color:{color};">{aqi:.2f}</p>
            <div class="aqi-category" style="color:{color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


#@st.cache_resource(show_spinner=False)
def build_circuit_figure(dynamic_inputs, n_qubits: int = 7):
    """
    Build a matplotlib figure of a mock 7-qubit variational circuit
    (AngleEmbedding + BasicEntanglerLayers), mirroring the backend model's
    architecture. Uses qml.draw_mpl if PennyLane is available, otherwise
    falls back to a hand-drawn matplotlib schematic.
    """
    if PENNYLANE_AVAILABLE:
        dev = qml.device("default.qubit", wires=n_qubits)
        weight_shape = qml.BasicEntanglerLayers.shape(n_layers=2, n_wires=n_qubits)
        rng = np.random.default_rng(42)
        weights = rng.uniform(0, np.pi, size=weight_shape)
        
        # Convert dynamic_inputs to numpy array and pad to 7 qubits
        dyn_arr = np.array(dynamic_inputs, dtype=float)
        if len(dyn_arr) < n_qubits:
            inputs = np.pad(dyn_arr, (0, n_qubits - len(dyn_arr)), 'constant')
        else:
            inputs = dyn_arr[:n_qubits]

        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # `level="device"` forces PennyLane to fully DECOMPOSE the high-level
        # templates (AngleEmbedding, BasicEntanglerLayers) down to the actual
        # fundamental gates the device executes — individual RX / RZ rotations
        # and the CNOT entangling ring — instead of drawing one solid block
        # per template. Older PennyLane versions (<0.33) don't accept a
        # `level` kwarg, so we fall back gracefully if it's unsupported.
        try:
            fig, ax = qml.draw_mpl(
                circuit,
                style="black_white_dark",   # dark-native palette, matches the UI
                level=2, 
                decimals=2,            
                fontsize=11,
            )(inputs, weights)
        except TypeError:
            # Fallback for older PennyLane: try the numeric level, then none.
            try:
                fig, ax = qml.draw_mpl(circuit, style="black_white_dark", level=2)(inputs, weights)
            except TypeError:
                fig, ax = qml.draw_mpl(circuit, style="black_white_dark")(inputs, weights)

        # Force transparent/dark background so it blends into the glass card,
        # regardless of what the chosen style already applied.
        fig.patch.set_facecolor("#060912")
        for a in fig.axes:
            a.set_facecolor("#060912")
        fig.tight_layout()
        return fig

    # ---- Fallback hand-drawn schematic if PennyLane isn't installed ----
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#060912")
    ax.set_facecolor("#060912")
    for q in range(n_qubits):
        y = n_qubits - q
        ax.hlines(y, 0, 10, color="#3a5a80", linewidth=1)
        ax.text(-0.4, y, f"|q{q}⟩", color="#00f5ff", fontsize=11, va="center", ha="right")
        ax.add_patch(plt.Rectangle((1.2, y - 0.25), 0.9, 0.5, color="#00c2ff", alpha=0.85))
        ax.text(1.65, y, "RX", color="#001018", fontsize=8, ha="center", va="center", weight="bold")
        ax.add_patch(plt.Rectangle((3.2, y - 0.25), 0.9, 0.5, color="#00ffb3", alpha=0.85))
        ax.text(3.65, y, "RY", color="#001018", fontsize=8, ha="center", va="center", weight="bold")
    for q in range(n_qubits - 1):
        y1, y2 = n_qubits - q, n_qubits - q - 1
        ax.plot([5.5, 5.5], [y1, y2], color="#ff5da2", linewidth=1.5)
        ax.scatter([5.5, 5.5], [y1, y2], color="#ff5da2", s=30, zorder=5)
    ax.text(5.5, n_qubits + 0.6, "Entangling Layer (CNOT ring)", color="#ff5da2",
            fontsize=9, ha="center")
    for q in range(n_qubits):
        y = n_qubits - q
        ax.add_patch(plt.Rectangle((7.6, y - 0.25), 0.9, 0.5, color="#ffd166", alpha=0.85))
        ax.text(8.05, y, "⟨Z⟩", color="#001018", fontsize=8, ha="center", va="center", weight="bold")
    ax.set_xlim(-1.2, 9.5)
    ax.set_ylim(0.2, n_qubits + 1.2)
    ax.axis("off")
    ax.set_title("7-Qubit Variational Circuit (AngleEmbedding + BasicEntanglerLayers)",
                 color="#e6f1ff", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def build_bloch_sphere(theta: float = 0.9, phi: float = 1.4):
    """Build an interactive 3D Bloch sphere with a state vector, using plotly."""
    # Sphere surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, "rgba(0,120,180,0.15)"], [1, "rgba(0,245,255,0.15)"]],
            showscale=False,
            opacity=0.35,
            contours=dict(
                x=dict(show=False), y=dict(show=False), z=dict(show=False)
            ),
        )
    )

    # Axes (X, Y, Z) through the sphere
    axis_len = 1.3
    for vec, name, color in [
        ([axis_len, 0, 0], "X", "#7f9bbf"),
        ([0, axis_len, 0], "Y", "#7f9bbf"),
        ([0, 0, axis_len], "Z", "#7f9bbf"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[-vec[0], vec[0]], y=[-vec[1], vec[1]], z=[-vec[2], vec[2]],
            mode="lines", line=dict(color=color, width=3), showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[vec[0]], y=[vec[1]], z=[vec[2]], mode="text",
            text=[name], textfont=dict(color="#e6f1ff", size=14), showlegend=False,
        ))

    # State vector |ψ⟩ = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)

    fig.add_trace(go.Scatter3d(
        x=[0, sx], y=[0, sy], z=[0, sz],
        mode="lines+markers",
        line=dict(color="#00ffb3", width=8),
        marker=dict(size=[0, 8], color="#00ffb3"),
        name="|ψ⟩ qubit state",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False, range=[-1.4, 1.4]),
            yaxis=dict(visible=False, range=[-1.4, 1.4]),
            zaxis=dict(visible=False, range=[-1.4, 1.4]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        showlegend=False,
    )
    return fig


def build_announcement_text(aqi: float) -> str:
    """Compose the sentence the browser will speak for a given AQI result."""
    label, _, _ = get_aqi_category(aqi)
    prefix = "Warning. " if aqi > 150 else "Inference complete. "
    return f"{prefix}The predicted air quality index is {aqi:.0f}, which is {label}."


def speak_announcement(text: str, run_id: int):
    """
    Inject a tiny, invisible HTML component that speaks `text` aloud using the
    browser-native Web Speech API (SpeechSynthesisUtterance) — no server-side
    TTS engine or extra Python dependency required.

    `run_id` is embedded in the component so that a *new* prediction with the
    exact same wording (e.g. two identical AQI results in a row) still forces
    Streamlit to re-mount the component and re-trigger speech, instead of
    silently reusing the previous iframe.
    """
    safe_text = text.replace("\\", "").replace('"', "'")
    components.html(
        f"""
        <script>
            (function() {{
                try {{
                    const synth = window.speechSynthesis;
                    synth.cancel();  // stop any previous utterance first
                    const utterance = new SpeechSynthesisUtterance("{safe_text}");
                    utterance.rate = 0.95;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    utterance.lang = "hi-IN";
                    synth.speak(utterance);
                }} catch (err) {{
                    console.warn("Speech synthesis unavailable:", err);
                }}
            }})();
        </script>
        <!-- run_id: {run_id} -->
        """,
        height=0,
        width=0,
    )


def call_predict_api(payload: dict, timeout: float = 120.0):
    """
    Call the FastAPI backend. Returns (result_dict_or_None, error_message_or_None).
    Handles connection failure, timeouts, and bad HTTP status codes gracefully.
    """
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            "🔌 **Can't reach the backend.** The FastAPI server doesn't seem to be "
            f"running at `{API_URL}`. Start it locally (e.g. `uvicorn main:app --reload`) "
            "and try again."
        )
    except requests.exceptions.Timeout:
        return None, "⏱️ The backend took too long to respond. Please try again."
    except requests.exceptions.HTTPError as e:
        return None, f"⚠️ Backend returned an error: `{e}`"
    except requests.exceptions.RequestException as e:
        return None, f"⚠️ Unexpected request error: `{e}`"
    except ValueError:
        return None, "⚠️ Backend returned a response that wasn't valid JSON."


# ----------------------------------------------------------------------------
# SIDEBAR — Inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="section-header">📍 Location & Context</p>', unsafe_allow_html=True)
    # 35+ Indian Cities with focus on UP and Bihar
    indian_cities = {
        # Bihar 
        "Patna": (25.5941, 85.1376),
        "Gopalganj": (26.4691, 83.9822),
        "Siwan": (26.2196, 84.3567),
        "Muzaffarpur": (26.1209, 85.3647),
        "Gaya": (24.7914, 85.0002),
        "Bhagalpur": (25.2425, 87.0126),
        "Darbhanga": (26.1542, 85.8918),
        "Purnia": (25.7711, 87.4753),
        "Begusarai": (25.4167, 86.1333),
        "Arrah": (25.5560, 84.6603),
        "Chhapra": (25.7796, 84.7499),
        
        # Uttar Pradesh
        "Kanpur": (26.4499, 80.3319),
        "Lucknow": (26.8467, 80.9462),
        "Varanasi": (25.3176, 82.9739),
        "Agra": (27.1767, 78.0081),
        "Prayagraj": (25.4358, 81.8463),
        "Meerut": (28.9845, 77.7064),
        "Gorakhpur": (26.7606, 83.3732),
        "Bareilly": (28.3670, 79.4304),
        "Aligarh": (27.8974, 78.0880),
        "Moradabad": (28.8386, 78.7733),
        "Jhansi": (25.4484, 78.5685),
        "Mathura": (27.4924, 77.6737),
        #"Noida": (28.5355, 77.3910)
        
        # Other Major Indian Cities
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bengaluru": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
        "Hyderabad": (17.3850, 78.4867),
        "Ahmedabad": (23.0225, 72.5714),
        "Pune": (18.5204, 73.8567),
        "Jaipur": (26.9124, 75.7873),
        "Surat": (21.1702, 72.8311),
        "Bhopal": (23.2599, 77.4126),
        "Indore": (22.7196, 75.8577),
        "Chandigarh": (30.7333, 76.7794)
    }

    # Sort the dictionary alphabetically so it looks clean in the dropdown
    sorted_cities = dict(sorted(indian_cities.items()))

    # Dropdown UI for the user
    selected_city = st.selectbox("Select City", options=list(sorted_cities.keys()))

    # Model inference ke liye coordinates extract karein (User ko bina dikhaye)
    latitude = sorted_cities[selected_city][0]
    longitude = sorted_cities[selected_city][1]
    live_weather = get_live_weather(latitude, longitude)
    
    
        
    # Optional caption
    st.caption(f"Coordinates: {latitude:.4f} N, {longitude:.4f} E")

    st.markdown("---")
    # 1. Season Dropdown
    season_name = st.selectbox(
        "Season", 
        options=["summer", "winter", "monsoon", "post_monsoon"],
        format_func=lambda s: s.replace("_", " ").title()
    )

    # 2. Day Type aur Crop Burning ek sath (Columns mein)
    col_day, col_crop = st.columns(2)
    with col_day:
        is_weekend_label = st.radio("Day Type", options=["Weekday", "Weekend"], horizontal=True)
        is_weekend = 1 if is_weekend_label == "Weekend" else 0
    with col_crop:
        st.write("") 
        crop_burning_season = 1 if st.toggle("🔥 Crop Burning", value=False) else 0 

    # 3. Voice Toggle
    voice_enabled = st.toggle("🔊 Voice Alerts", value=True, help="Speaks the AQI result aloud")

    st.markdown("---")
    predict_clicked = st.button("☁️ Run Quantum Inference", use_container_width=True, type="primary")

    st.markdown(
        """
        <div class="footer-note">
        Backend: FastAPI · Model: Hybrid QNN<br>
        PyTorch + PennyLane · 7 Qubits
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="quantum-title">⚛️ Quantum AQI Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="quantum-subtitle">Hybrid Quantum-Classical Neural Network · Air Quality Inference Engine</p>',
    unsafe_allow_html=True,
)
st.write("")

# ==========================================
# MAIN SCREEN - LIVE WEATHER BANNER
# ==========================================
if live_weather:
    st.markdown('<p class="section-header" style="margin-top: 10px;">🌤️ LIVE WEATHER CONTEXT</p>', unsafe_allow_html=True)
    
    # 3 Metrics ko horizontally dikhane ke liye
    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Temperature", f"{live_weather['temperature']} °C")
    wc2.metric("Humidity", f"{live_weather['humidity']} %")
    wc3.metric("Wind Speed", f"{live_weather['wind_speed']} m/s")
    
    st.markdown("---")

# Keep the last prediction across reruns (e.g. when just tweaking the Bloch sphere)
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "voice_run_id" not in st.session_state:
    st.session_state.voice_run_id = 0

if predict_clicked:
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "is_weekend": is_weekend,
        "crop_burning_season": crop_burning_season,
        "season_name": season_name
        
    }
    


    with st.spinner("Encoding features into quantum states and running the circuit…"):
        # --- AGGRESSIVE MOBILE SIDEBAR CLOSE ---
        components.html(
            f"""
            <script>
                // 500ms ka delay taaki spinner load hone ke baad close command chale
                setTimeout(() => {{
                    const doc = window.parent.document;
                    
                    // Trick 1: Keyboard se 'Escape' button dabana (Mobile par sabse effective)
                    doc.dispatchEvent(new KeyboardEvent('keydown', {{'key': 'Escape', 'bubbles': true}}));
                    
                     // Trick 2: Sidebar ke bahar background overlay par click karna
                    const appContainer = doc.querySelector('[data-testid="stAppViewContainer"]');
                    if (appContainer) {{
                        appContainer.click();
                    }}
                }}, 500); 
            </script>
            <div style="display:none;">{time.time()}</div>
            """,
            height=0,
            width=0
        )
        
        result, error = call_predict_api(payload)
        
    st.session_state.last_result = result
    st.session_state.last_error = error
    st.session_state.voice_run_id += 1  # forces the voice component to re-fire


# ----------------------------------------------------------------------------
# TOP ROW — AQI Result + Request Summary
# ----------------------------------------------------------------------------
col_result, col_summary = st.columns([1.1, 1.4], gap="large")

with col_result:
    st.markdown('<p class="section-header">🌫️ Prediction Result</p>', unsafe_allow_html=True)
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    elif st.session_state.last_result and st.session_state.last_result.get("status") == "success":
        aqi_value = st.session_state.last_result["predicted_aqi"]
        render_aqi_display(aqi_value)
        
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = aqi_value,
            title = {'text': "AQI Status", 'font': {'size': 18, 'color': '#00f5ff'}},
            gauge = {
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "rgba(255,255,255,0.7)"}, # Indicator bar ki transparency
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 1,
                'bordercolor': "#3a5a80",
                'steps': [
                    {'range': [0, 50], 'color': "#00ff00"},       # Good (Green)
                    {'range': [50, 100], 'color': "#ffff00"},     # Moderate (Yellow)
                    {'range': [100, 150], 'color': "#ffa500"},    # Unhealthy for Sensitive (Orange)
                    {'range': [150, 200], 'color': "#ff0000"},    # Unhealthy (Red)
                    {'range': [200, 300], 'color': "#800080"},    # Very Unhealthy (Purple)
                    {'range': [300, 500], 'color': "#800000"}     # Hazardous (Maroon)
                ]
            }
        ))
        
        # Transparent background set karna taaki aapke dark UI me blend ho jaye
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250)
        
        # Chart ko frontend pe render karna
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"📡 {st.session_state.last_result.get('message', '')}")

        # 🔊 Voice feedback — only fires right after a fresh prediction,
        # driven off `voice_run_id` so it doesn't replay on unrelated reruns
        # (e.g. dragging the Bloch sphere sliders).
        if voice_enabled and predict_clicked:
            # AQI Category ka logic
            if aqi_value <= 50:
                category = "Good"
            elif aqi_value <= 100:
                category = "Moderate"
            elif aqi_value <= 150:
                category = "Unhealthy for Sensitive Groups"
            elif aqi_value <= 200:
                category = "Unhealthy"
            elif aqi_value <= 300:
                category = "Very Unhealthy"
            else:
                category = "Hazardous"
                
            # Text English me rahega, par previous settings (en-IN) ki wajah se Indian voice me aayega
            hindi_text= f"predicted A Q I {aqi_value} hai . jo ki {category} category me aata hai."
            
            speak_announcement(hindi_text, st.session_state.voice_run_id)
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; opacity:0.7;">
                Configure your inputs in the sidebar, then hit
                <b>“Run Quantum Inference”</b> to get a live AQI prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_summary:
    st.markdown('<p class="section-header">🧾 Request Payload</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    payload_preview = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "is_weekend": is_weekend,
        "crop_burning_season": crop_burning_season,
        "season_name": season_name
        
    }
    st.json(payload_preview)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"Target endpoint: `{API_URL}`")


st.write("")
st.markdown("---")

# ----------------------------------------------------------------------------
# BOTTOM ROW — Quantum Visualizations
# ----------------------------------------------------------------------------
col_circuit, col_bloch = st.columns([1.3, 1], gap="large")

with col_circuit:
    st.markdown('<p class="section-header">🔗 Model Circuit Architecture</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Real slider values ka list bana kar pass karein
    current_inputs = [latitude, longitude, float(is_weekend), float(crop_burning_season)]
    fig = build_circuit_figure(current_inputs)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Mock visualization of the backend's variational circuit: 7 qubits, "
        "AngleEmbedding for classical-to-quantum feature encoding, followed by "
        "BasicEntanglerLayers for trainable entanglement."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_bloch:
    st.markdown('<p class="section-header">🌐 Qubit State — Bloch Sphere</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        theta = st.slider("θ (polar)", 0.0, float(np.pi), 0.9, 0.05, key="theta_slider")
    with b_col2:
        phi = st.slider("φ (azimuthal)", 0.0, float(2 * np.pi), 1.4, 0.05, key="phi_slider")
    bloch_fig = build_bloch_sphere(theta, phi)
    st.plotly_chart(bloch_fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Drag to rotate. The green vector represents a single qubit's state |ψ⟩.")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="footer-note">
    Built for demonstration purposes · Connects to a locally-hosted FastAPI + PennyLane backend
    </div>
    """,
    unsafe_allow_html=True,
)
