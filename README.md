
# AI-Driven Predictive Maintenance System

This project implements a full **AI-Driven Predictive Maintenance** pipeline, designed to simulate real-time engine sensor data, detect anomalies, predict failure risks, and visualize system health through an interactive dashboard.

## 🚀 Project Overview

The system consists of:
- **IEP #1**: Engine Failure Risk Prediction (Binary Classifier - Neural Network)
- **IEP #2**: Sensor Anomaly Detection (Autoencoder)
- **EEP Backend**: Central logging and coordination
- **EEP Dashboard**: Real-time status visualization using Streamlit
- **Data Streamer**: Simulates live sensor data feed
- **Docker Compose**: Orchestrates all services

---

## 📂 Project Structure

```
ai-driven-pred-maint/
├── docker-compose.yml
├── data_streamer/                  # Real-time data streaming service
├── iep1_failuredetection/          # IEP #1 - Failure Risk Prediction API + Model
├── iep2_anomalydetection/          # IEP #2 - Anomaly Detection API + Model
├── eep_backend/                    # Backend for logging system status
├── eep_dashboardUI/                # Streamlit Dashboard for visualization
├── datasets/                       # (Git-ignored) Engine sensor dataset
└── shared_log/                     # Shared volume for logging (eep_data_log.csv)
```

---

## ⚙️ How to Run the System

Ensure you have **Docker** and **Docker Compose** installed.

1. Clone the repository.
2. Place `engine_failure_dataset.csv` in the `datasets/` folder.
3. Run the entire system with:

```bash
docker-compose up --build
```

Access the dashboard at: `http://localhost:8501`

---

## 🧩 Components Breakdown

### 1️⃣ IEP #1 - Failure Detection (`iep1_failuredetection/`)
- Flask API serving a Neural Network classifier.
- Endpoint: `POST /predict-risk`
- Predicts engine failure risk based on sensor data.

### 2️⃣ IEP #2 - Anomaly Detection (`iep2_anomalydetection/`)
- Flask API serving an Autoencoder for anomaly detection.
- Endpoint: `POST /ingest-sensor-data`
- Identifies abnormal sensor behavior and flags top faulty sensors.

### 3️⃣ EEP Backend (`eep_backend/`)
- Central service that logs combined outputs from both IEPs.
- Endpoint: `POST /update-dashboard`
- Stores logs in `eep_data_log.csv` (shared volume).

### 4️⃣ EEP Dashboard (`eep_dashboardUI/`)
- Streamlit dashboard that auto-refreshes to display system status.
- Visualizes risk levels, anomalies, flagged sensors, and timestamps.

### 5️⃣ Data Streamer (`data_streamer/`)
- Simulates real-time sensor data by sending entries to IEPs.
- Aggregates responses and updates the EEP backend.

---

## 🌐 API Endpoints Summary

| Service         | Endpoint               | Method | Description                  |
|-----------------|------------------------|--------|------------------------------|
| IEP #1          | `/predict-risk`        | POST   | Predict failure risk         |
| IEP #2          | `/ingest-sensor-data`  | POST   | Detect anomalies             |
| EEP Backend     | `/update-dashboard`    | POST   | Log system status            |
| All Services    | `/health`              | GET    | Health check endpoints       |

---

## 📊 Example Dashboard Output
- **Status**: 🟢 Healthy / 🟡 Attention Needed / 🔴 Immediate Action Required
- **Displays**:
  - Risk Level & Probability
  - Anomaly Status
  - Flagged Sensors
  - Last Update Timestamp

---

## 📦 Requirements
> No manual setup needed due to Docker.

For local development:
- Python 3.x
- Flask
- TensorFlow
- Scikit-learn
- Streamlit
- Requests
- Joblib

---

## 🚨 Notes
- The `datasets/` folder is **git-ignored** for data privacy.
- This system is intended for demonstration and prototyping purposes.
- For production use, further enhancements in security, scalability, and monitoring are recommended.

---

## 📈 Next Steps
- Integrate LLM-based chatbot for advanced diagnostics.
- Enhance dashboard with historical trends and analytics.
