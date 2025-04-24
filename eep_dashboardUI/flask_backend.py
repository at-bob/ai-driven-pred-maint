from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)
log_file = "eep_data_log.csv"

# Initialize log if not exists
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        'timestamp', 'risk_status', 'risk_probability',
        'anomaly_status', 'reconstruction_error', 'flagged_sensors'
    ]).to_csv(log_file, index=False)

@app.route('/update-dashboard', methods=['POST'])
def update_dashboard():
    data = request.get_json()
    pd.DataFrame([data]).to_csv(log_file, mode='a', header=False, index=False)
    return jsonify({"status": "Dashboard updated"}), 200

if __name__ == "__main__":
    app.run(port=5003)
