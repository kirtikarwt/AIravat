# AIravat — Extreme-Weather Anomaly Intelligence

> **SIH 2026 · Problem Statement 26078 · Ministry of Earth Sciences (MoES)**  
> *AI-driven spatio-temporal tracking of extreme weather anomalies in medium-range forecasts.*  
> *"Days ahead · kilometres close"*

---

## 🌪️ Overview

Standard numerical weather prediction runs (like NCMRWF's 23-member NEPS-G ensemble) operate at ~12 km resolution. While effective at synoptic scales, a 12 km grid cannot resolve localized convective bursts, severe cloudbursts, or orographic flood triggers.

**AIravat** solves this by:
1. **Finding the Extreme Early**: Ingests ensemble forecasts, computing exact closed-form discrete Lalaurette **Extreme Forecast Index (EFI)** and **Shift of Tails (SOT)** against a 30-year local climatology (ERA5 baseline).
2. **Dynamic Anomaly Tracking**: Uses spherical Haversine DBSCAN clustering to isolate **640 × 640 km danger zones**.
3. **Physics-Constrained Downscaling**: Zooms selectively into identified danger zones from 12 km to **5 km** using conditional generative diffusion, verified with test-time mass/moisture divergence projections.
4. **Actionable Plain-Language Alerts**: Provides district collectors and NDMA/SDRF with clear, jargon-free alerts, audit trails, and conversational intelligence in English and Hindi.

---

## ✨ Features & Dashboard Pages

- **Tactical Overview**: Interactive satellite map with EFI/SOT heatmaps, active cyclone trajectories, and real-time Event Intelligence cards.
- **Live Pipeline Feed**: Step-by-step ingestion, quality check, anomaly detection, diffusion downscaling, and CAP alert publishing replay.
- **Anomaly Tracking**: 10-day temporal evolution charts, cyclone central pressure, track uncertainty cones, and bounding box tables.
- **5 km Downscaling**: Direct visual comparison (12 km vs MSE vs AIravat generative vs truth) with radially-averaged power spectral density (PSD) preservation ($k^{-5/3}$).
- **Pinpoint Alerts**: City-level and coordinate-level diagnostic lookup for over 30 Indian cities with ensemble distribution histograms.
- **Alert Reasons & Audit Trail**: Full transparent score build-up waterfall, rule-check matrix, climatology comparison, and downloadable JSON audit trails.
- **AIravat Copilot**: Conversational AI assistant supporting English and Hindi for instant situational reports and location queries.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 & 3.13)
- `pip`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/kirtikarwt/AIravat.git
cd AIravat

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
AIravat/
├── .streamlit/
│   └── config.toml          # Streamlit configuration & static serving
├── src/
│   └── dashboard/
│       ├── app.py           # Main Streamlit dashboard application
│       ├── engine.py        # Core analytics (EFI, SOT, DBSCAN, downscaling demo)
│       ├── assets/          # Media, branding & tech docs
│       └── static/          # Static media for browser streaming
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 📜 License
MIT License
