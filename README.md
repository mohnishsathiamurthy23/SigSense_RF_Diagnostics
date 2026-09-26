# 📡 SigSense RF Diagnostics

> **Domain-Informed, Microsecond AI for Real-Time RF Signal Intelligence**

SigSense is a lightweight software-based RF diagnostic terminal designed to classify signal noise regimes and evaluate channel impairments in real time without the compute overhead or latency of heavy deep neural networks.

---

## 🚀 Key Novelty & Architectural Advantages

* **Physics-Guided Feature Compression:** Transforms raw 128-sample complex $I/Q$ frames into 383 dense physical signatures (Normalized PSD, Phase Step Differentials $\Delta\phi$, and Envelope Magnitudes).
* **Microsecond CPU Inference (~10 µs):** Replaces slow matrix-multiplication deep learning models with an optimized LightGBM ensemble, preventing streaming buffer overruns on standard host CPUs.
* **Operational Regime Classification:** Classifies signals into three actionable noise states:
  * **Low Noise Regime** (≥ +6 dB)
  * **Medium Noise Regime** (0 dB to +5 dB)
  * **High Noise Regime** (< 0 dB)
* **Interpretable Signal Diagnostics:** Pairs machine learning outputs with real-time physical metrics (Signal RMS Voltage, PAPR, Phase Variance, and Dominant Frequency Bins).

---

## 📊 Dataset & Benchmark

Evaluated on **RML2016.10a**, the gold-standard benchmark dataset for machine learning in radio frequency signal analysis.
* Filtered to remove silent analog modulations and severe sub--10 dB noise floor artifacts.
* Validated across digital modulation schemes (e.g., 8PSK, QAM16, QPSK, BPSK) under realistic simulated multipath fading and carrier offsets.

---

## 📦 Dataset Setup & File Placement

> **Repository File Note:** Due to GitHub's file size restrictions, the pre-processed benchmark dataset (`RML2016.10a_dict_optimized.pkl`) could not be uploaded directly to the repository.

1. **Download the Dataset File:**  
   📥 **[Click Here to Download `RML2016.10a_dict_optimized.pkl`]((https://drive.google.com/file/d/1RnSuMlVYktrFmZpQ4kYDAMY04eZc1HDF/view?usp=sharing))**

2. **Place File in Project Directory:**  
   Save the downloaded `.pkl` file directly inside the root folder alongside `app.py`:

```text
SigSense_RF_Diagnostics/
├── app.py
├── RML2016.10a_dict_optimized.pkl   <-- Place downloaded file here
├── requirements.txt
├── LICENSE
└── README.md
