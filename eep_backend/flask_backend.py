from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)
log_file = "/app/log/eep_data_log.csv"

# Initialize log if not exists
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        'timestamp', 'risk_status', 'risk_probability',
        'anomaly_status', 'flagged_sensors'
    ]).to_csv(log_file, index=False)

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/update-dashboard', methods=['POST'])
def update_dashboard():
    data = request.get_json()
    print(f"Received data: {data}")   # Add this line
    try:
        pd.DataFrame([data]).to_csv(log_file, mode='a', header=False, index=False)
        print("Data logged successfully.")
    except Exception as e:
        print(f"Logging failed: {e}")
    return jsonify({"status": "Dashboard updated"}), 200
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)
