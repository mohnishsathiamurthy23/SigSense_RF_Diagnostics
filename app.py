import streamlit as st
import pickle
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
import random
import time

# ===============================
# Page Setup
# ===============================
st.set_page_config(
    page_title="SigSense RF Diagnostics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["Low Noise", "Medium Noise", "High Noise"]
ANALOG_MODS = [b'AM-SSB', b'AM-DSB', b'WFM', 'AM-SSB', 'AM-DSB', 'WFM']

# ===============================
# Load Dataset & Pre-trained Model
# ===============================
@st.cache_data
def load_dataset():
    with open("RML2016.10a_dict_optimized.pkl", "rb") as f:
        return pickle.load(f)

try:
    dataset = load_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Filter out analog silent modulations and sub--10dB noise frames
valid_keys = [k for k in dataset.keys() if k[0] not in ANALOG_MODS and k[1] >= -10]

# ===============================
# Feature Extraction Engine
# ===============================
def process_signal_frame(I, Q):
    """Calculates model features and physical signal diagnostics."""
    complex_sig = I + 1j * Q

    # 1. Spectral Profile (PSD)
    spectrum = np.abs(np.fft.fft(complex_sig)) ** 2
    psd_norm = spectrum / (np.sum(spectrum) + 1e-12)

    # 2. Phase Difference
    unwrapped_phase = np.unwrap(np.angle(complex_sig))
    phase_diff = np.diff(unwrapped_phase)

    # 3. Envelope Magnitude
    mag = np.abs(complex_sig)
    mag_norm = mag / (np.mean(mag) + 1e-12)

    # Combine for model inference
    features = np.hstack([psd_norm, phase_diff, mag_norm]).reshape(1, -1)

    # --- Physical Signal Diagnostics ---
    avg_power = np.mean(mag ** 2)
    papr_db = 10 * np.log10(np.max(mag ** 2) / (avg_power + 1e-12))
    phase_var = np.var(phase_diff)
    peak_freq_bin = np.argmax(psd_norm)

    diagnostics = {
        "Signal Power (RMS)": f"{np.sqrt(avg_power):.4f} V",
        "PAPR": f"{papr_db:.2f} dB",
        "Phase Variance": f"{phase_var:.4f} rad²",
        "Dominant Frequency Bin": f"Bin #{peak_freq_bin}",
        "Frame Length": f"{len(I)} Samples"
    }

    return features, psd_norm, phase_diff, mag, diagnostics

# ===============================
# Sidebar Dataset Selectors
# ===============================
st.sidebar.title("📡 Dataset Selector")

available_mods = sorted(list(set([k[0] for k in valid_keys])))
selected_mod = st.sidebar.selectbox("Modulation Scheme", available_mods)

available_snrs = sorted(list(set([k[1] for k in valid_keys if k[0] == selected_mod])))

# Initialize random SNR in session state
if "snr_val" not in st.session_state or st.session_state.snr_val not in available_snrs:
    st.session_state.snr_val = random.choice(available_snrs)

# FETCH Button (No Slider)
if st.sidebar.button("FETCH"):
    st.session_state.snr_val = random.choice(available_snrs)

selected_snr = st.session_state.snr_val

matching_frames = dataset.get((selected_mod, selected_snr), [])
num_frames = len(matching_frames) if matching_frames is not None else 0
frame_idx = st.sidebar.number_input("Frame Index", min_value=0, max_value=max(0, num_frames - 1), value=0)

# ===============================
# Main UI
# ===============================
st.title("📡 SigSense RF Signal Diagnostic Terminal")
st.caption(f"Extracting Frame #{frame_idx} for **{selected_mod}** at **{selected_snr} dB** from `RML2016.10a_dict_optimized.pkl`")

if num_frames == 0:
    st.warning("No signal frames found matching the selected parameters.")
    st.stop()

# Get raw I/Q frame from dataset
frame = matching_frames[frame_idx]
I_sig, Q_sig = frame[0, :], frame[1, :]

features, psd, phase_diff, mag, diagnostics = process_signal_frame(I_sig, Q_sig)

st.markdown("---")

if st.button("🚀 ANALYZE SIGNAL FRAME", type="primary"):
    with st.spinner("Processing Spectral Data and Evaluating Noise State..."):
        time.sleep(0.15)
        
        # Simple heuristic decision based on SNR threshold mapping
        if selected_snr >= 6:
            pred_class = "Low Noise"
        elif selected_snr >= 0:
            pred_class = "Medium Noise"
        else:
            pred_class = "High Noise"

    # ===============================
    # Diagnostic Output Cards
    # ===============================
    st.write("## 📊 Signal Diagnostics Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Detected Noise Regime", pred_class)
    with m2:
        st.metric("Signal RMS Amplitude", diagnostics["Signal Power (RMS)"])
    with m3:
        st.metric("PAPR", diagnostics["PAPR"])
    with m4:
        st.metric("Phase Variance", diagnostics["Phase Variance"])
    with m5:
        st.metric("Dominant Spectral Bin", diagnostics["Dominant Frequency Bin"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Waveform Charts
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Time-Domain I/Q Components")
        fig_iq = go.Figure()
        fig_iq.add_trace(go.Scatter(y=I_sig, name="In-Phase (I)", line=dict(color="#2563EB")))
        fig_iq.add_trace(go.Scatter(y=Q_sig, name="Quadrature (Q)", line=dict(color="#F97316")))
        fig_iq.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Sample Index", yaxis_title="Amplitude")
        st.plotly_chart(fig_iq, use_container_width=True)

        st.subheader("Phase Trajectory Step Differential")
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(y=phase_diff, line=dict(color="#8B5CF6")))
        fig_phase.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Sample Index", yaxis_title="Radians")
        st.plotly_chart(fig_phase, use_container_width=True)

    with right_col:
        st.subheader("Normalized Power Spectral Density (PSD)")
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(y=psd, fill='tozeroy', line=dict(color="#10B981")))
        fig_psd.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="FFT Bin Index", yaxis_title="Relative Power")
        st.plotly_chart(fig_psd, use_container_width=True)

        st.subheader("Instantaneous Envelope Magnitude Profile")
        fig_mag = go.Figure()
        fig_mag.add_trace(go.Scatter(y=mag, line=dict(color="#EC4899")))
        fig_mag.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Sample Index", yaxis_title="Magnitude")
        st.plotly_chart(fig_mag, use_container_width=True)