from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import json

# Load sensor names
with open("model/sensor_names.json", "r") as f:
    sensor_names = json.load(f)

# Load model and components
model = tf.keras.models.load_model("model/autoencoder_model.h5", compile = False)
scaler = joblib.load("model/scaler.pkl")
with open("model/threshold.txt", "r") as f:
    threshold = float(f.read())

app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "IEP #2 Anomaly Detection API is running."})

# Main prediction endpoint
@app.route('/ingest-sensor-data', methods=['POST'])
def ingest_sensor_data():
    try:
        data = request.get_json()
        sensor_values = data.get('sensor_values', None)

        if sensor_values is None:
            return jsonify({"error": "Missing 'sensor_values'"}), 400

        # Handle Operational Mode Mapping
        mode_map = {'Idle': 1, 'Cruising': 2, 'Heavy Load': 3}
        sensor_array = np.array(sensor_values, dtype=object).reshape(1, -1)

        if isinstance(sensor_array[0, -1], str):
            sensor_array[0, -1] = mode_map.get(sensor_array[0, -1], 0)

        sensor_array = sensor_array.astype(float)

        # Scale input
        scaled_input = scaler.transform(sensor_array)

        # Predict reconstruction
        reconstructed = model.predict(scaled_input)

        # Calculate per-sensor reconstruction error
        per_sensor_error = np.square(scaled_input - reconstructed)[0]

        # Overall reconstruction error
        overall_error = np.mean(per_sensor_error)

        # Dynamic threshold (if applied)
        live_threshold = threshold * 3

        if overall_error > live_threshold:
            severity = "Anomaly Detected"
            # Identify top 2 faulty sensors
            top_sensors_idx = np.argsort(per_sensor_error)[-2:]
            flagged_sensors = [sensor_names[j] for j in top_sensors_idx]
        else:
            severity = "Normal"
            flagged_sensors = []

        return jsonify({
            "status": severity,
            "reconstruction_error": round(overall_error, 5),
            "flagged_sensors": flagged_sensors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
