
# AI-Driven Predictive Maintenance

This project implements an end-to-end AI-driven predictive maintenance system designed to simulate real-time engine sensor data, detect anomalies, and predict engine failure risks.

## Project Structure

```
ai-driven-pred-maint/
├── README.md
├── anomalydetection.py          # Script to train Autoencoder, find threshold, and save model artifacts
├── data_streamer.py             # Real-time data streamer to simulate sensor data flow
├── engine_failure_dataset.csv   # Dataset containing historical engine sensor data
└── iep2_anomalydetection/
    ├── app.py                   # Flask API for IEP #2 - Sensor Anomaly Detection
    ├── requirements.txt         # Dependencies for the Flask API
    └── model/
        ├── autoencoder_model.h5 # Trained Autoencoder model
        ├── scaler.pkl           # Scaler used for preprocessing
        └── threshold.txt        # Optimal threshold for anomaly detection
```

## Components

### 1. `anomalydetection.py`
- Trains an Autoencoder on normal engine behavior.
- Determines the optimal anomaly detection threshold.
- Saves the trained model, scaler, and threshold for deployment.

### 2. `iep2_anomalydetection/app.py`
- Flask API serving the anomaly detection model.
- Endpoint `/ingest-sensor-data` accepts raw sensor data and returns anomaly status.
- Dynamic thresholding applied for realistic streaming scenarios.

### 3. `data_streamer.py`
- Simulates real-time streaming by sending rows of sensor data to the IEP #2 API.
- Adjustable delay to mimic live data feed.

## How to Run

1. **Start the Flask API**
   ```bash
   cd iep2_anomalydetection
   python3 app.py
   ```

2. **Run the Data Streamer**
   In a separate terminal:
   ```bash
   python3 data_streamer.py
   ```

## API Endpoints

- `GET /health`  
  Check if the API is running.

- `POST /ingest-sensor-data`  
  Send sensor data and receive anomaly detection response.

## Example API Response
```json
{
  "anomaly": 0,
  "reconstruction_error": 0.00487,
  "live_threshold": 0.06412
}
```

## Next Steps
- Implement IEP #1 for engine failure risk prediction.
- Develop the External Endpoint (EEP) with an LLM chatbot and UI.
- Dockerize the entire system for deployment.

## Requirements
- Python 3.x
- Flask
- TensorFlow
- Scikit-learn
- Requests