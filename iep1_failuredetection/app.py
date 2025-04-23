from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

# === Load Model and Scaler ===
model = tf.keras.models.load_model("model/failure_classifier_binary.h5")
scaler = joblib.load("model/scaler.pkl")

app = Flask(__name__)

# === Health Check Endpoint ===
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "IEP #1 Failure Detection API is running."})

# === Prediction Endpoint ===
@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    try:
        data = request.get_json()
        sensor_values = data.get('sensor_values', None)

        if sensor_values is None:
            return jsonify({"error": "Missing 'sensor_values'"}), 400

        # Convert Operational_Mode if it's still string
        mode_map = {'Idle': 1, 'Cruising': 2, 'Heavy Load': 3}
        sensor_array = np.array(sensor_values, dtype=object).reshape(1, -1)

        if isinstance(sensor_array[0, -1], str):
            sensor_array[0, -1] = mode_map.get(sensor_array[0, -1], 0)

        sensor_array = sensor_array.astype(float)

        # Scale input
        scaled_input = scaler.transform(sensor_array)

        # Predict risk (returns probability)
        risk_prob = model.predict(scaled_input)[0][0]
        risk_flag = int(risk_prob >= 0.5)

        # Map output
        risk_status = "Risk of Failure" if risk_flag == 1 else "No Risk"

        return jsonify({
            "risk_flag": risk_flag,
            "risk_status": risk_status,
            "risk_probability": round(float(risk_prob), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
