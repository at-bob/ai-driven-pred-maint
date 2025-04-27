import pandas as pd
import time
import requests
import json
import datetime
import socket
import time

def wait_for_service(host, port, timeout=60):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"✅ Service {host}:{port} is up!")
                return
        except OSError:
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"⏰ Timeout: {host}:{port} not available after {timeout} seconds")
            print(f"⏳ Waiting for {host}:{port}...")
            time.sleep(2)

def wait_for_http_service(url, timeout=60):
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"✅ Service at {url} is up and responding!")
                return
        except requests.exceptions.ConnectionError:
            pass

        if time.time() - start_time >= timeout:
            raise TimeoutError(f"⏰ Timeout: {url} not available after {timeout} seconds")

        print(f"⏳ Waiting for {url}...")
        time.sleep(2)

if __name__ == "__main__":
    try:
        # Wait for services
        wait_for_service('iep1', 5001)
        wait_for_service('iep2', 5002)
        wait_for_service('eep_backend', 5003)
        wait_for_http_service("http://eep_backend:5003/health")

        print("🚀 All services are up. Starting data streaming...")

        # Load dataset
        df = pd.read_csv('datasets/engine_failure_dataset.csv')

        # Drop columns not needed for input (keep raw sensor data + Operational_Mode)
        input_columns = df.drop(columns=['Fault_Condition', 'Time_Stamp']).columns

        # EEP Backend URL
        EEP_BACKEND_URL = "http://eep_backend:5003/update-dashboard"

        # Simulate streaming
        for index, row in df.iterrows():
            sensor_data = row[input_columns].tolist()

            payload = {
                "sensor_values": sensor_data
            }

            try:
                # Send to IEP #1
                risk_response = requests.post("http://iep1:5001/predict-risk",
                                            headers={"Content-Type": "application/json"},
                                            data=json.dumps(payload))
                risk_output = risk_response.json()

                # Send to IEP #2
                anomaly_response = requests.post("http://iep2:5002/ingest-sensor-data",
                                                headers={"Content-Type": "application/json"},
                                                data=json.dumps(payload))
                anomaly_output = anomaly_response.json()

                # Combine outputs for EEP
                combined_output = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "risk_status": risk_output.get('risk_status'),
                    "risk_probability": risk_output.get('risk_probability'),
                    "anomaly_status": anomaly_output.get('status'),
                    "flagged_sensors": ", ".join(anomaly_output.get('flagged_sensors', []))
                }

                # Send to EEP backend
                eep_response = requests.post(EEP_BACKEND_URL,
                                            headers={"Content-Type": "application/json"},
                                            data=json.dumps(combined_output))

                if eep_response.status_code == 200:
                    print(f"Row {index} processed and sent to dashboard.")
                else:
                    print(f"EEP Error at row {index}: {eep_response.text}")

            except Exception as e:
                print(f"Error at row {index}: {e}")

            time.sleep(5)  # Simulate real-time delay

    except Exception as e:
        print(f"🔥 Fatal Error: {e}")