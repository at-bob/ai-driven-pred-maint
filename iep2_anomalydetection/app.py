from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

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
            return jsonify({"error": "Missing 'sensor_values' in request."}), 400
        
        mode_map = {'Idle': 1, 'Cruising': 2, 'Heavy Load': 3}

        sensor_array = np.array(sensor_values, dtype=object).reshape(1, -1)

        # Convert Operational_Mode if it's a string
        if isinstance(sensor_array[0, -1], str):
            sensor_array[0, -1] = mode_map.get(sensor_array[0, -1], 0)  # Default to 0 if unknown

        # Ensure all values are now floats (Operational_Mode is now numeric)
        sensor_array = sensor_array.astype(float)

        scaled_input = scaler.transform(sensor_array)
        reconstructed = model.predict(scaled_input)

        # Calculate reconstruction error
        error = np.mean(np.square(scaled_input - reconstructed), axis=1)[0]

        # Dynamic threshold adjustment
        live_threshold = threshold * 10

        is_anomaly = int(error > live_threshold)

        # Build response
        response = {
            "anomaly": is_anomaly,
            "reconstruction_error": round(error, 5),
            "live_threshold": round(live_threshold, 5)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
