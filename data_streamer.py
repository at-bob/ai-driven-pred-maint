import pandas as pd
import time
import requests
import json

# Load dataset
df = pd.read_csv('engine_failure_dataset.csv')

# Drop columns not needed for input (keep raw sensor data + Operational_Mode)
input_columns = df.drop(columns=['Fault_Condition', 'Time_Stamp']).columns

# Simulate streaming
for index, row in df.iterrows():
    # Extract raw sensor data as list
    sensor_data = row[input_columns].tolist()

    # Prepare JSON payload
    payload = {
        "sensor_values": sensor_data
    }

    try:
         # Send to IEP #1 (Failure Risk Detection)
        risk_response = requests.post("http://127.0.0.1:5001/predict-risk",
                                       headers={"Content-Type": "application/json"},
                                       data=json.dumps(payload))
        risk_output = risk_response.json()

        # Send to IEP #2 (Anomaly Detection)
        anomaly_response = requests.post("http://127.0.0.1:5002/ingest-sensor-data",
                                          headers={"Content-Type": "application/json"},
                                          data=json.dumps(payload))
        anomaly_output = anomaly_response.json()

        print(f"\nRow {index}:")
        print(f"IEP #1 - Risk Detection ➜ {risk_output}")
        print(f"IEP #2 - Anomaly Detection ➜ {anomaly_output}")

    except Exception as e:
        print(f"Error at row {index}: {e}")
    # Wait to simulate real-time (adjust as needed)
    time.sleep(5)