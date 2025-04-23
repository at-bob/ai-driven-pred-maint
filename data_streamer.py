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
        # Send to IEP #2
        response = requests.post("http://127.0.0.1:5002/ingest-sensor-data",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload))

        print(f"Row {index} ➜ IEP #2 Response: {response.json()}")

    except Exception as e:
        print(f"Error sending data at row {index}: {e}")

    # Wait to simulate real-time (adjust as needed)
    time.sleep(1)