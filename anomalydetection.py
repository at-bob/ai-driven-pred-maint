import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set working directory to where this script is located
os.chdir(os.path.dirname(__file__))

# Create model folder if not exists
os.makedirs("iep2_anomalydetection/model", exist_ok=True)

# Load dataset
df = pd.read_csv('engine_failure_dataset.csv', parse_dates=['Time_Stamp'])

# Set the timestamp as the index
df.set_index('Time_Stamp', inplace=True)

#extract the readings for severe faults
SevereFault= df[df['Fault_Condition']==3]
ModerateFault= df[df['Fault_Condition']==2]
MinorFault= df[df['Fault_Condition']==1]

#Extract the name of the numerical columns
df_wo_fault = df.drop(['Fault_Condition'],axis=1)
names= df_wo_fault.columns

# change the operational mode column to integer values 
mode_map = {
    'Idle': 1,
    'Cruising': 2,
    'Heavy Load': 3
}

df['Operational_Mode'] = df['Operational_Mode'].map(mode_map)

# Drop 'Fault_Condition' (target) 
# 'Operational_Mode' is already converted to numeric
df_features = df.drop(columns=['Fault_Condition'])

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_features), index=df_features.index, columns=df_features.columns)

true_fault_labels = df.loc[df.index.isin(df_scaled.index) & df['Fault_Condition'].isin([1, 2, 3]), 'Fault_Condition']

# Use only normal data (Fault_Condition == 0) to train the model
df_normal = df[df['Fault_Condition'] == 0].drop(columns=['Fault_Condition'])
df_anomaly_test = df[df['Fault_Condition'].isin([1, 2, 3])].drop(columns=['Fault_Condition'])

# Scale all data
scaler = StandardScaler()
X_train = scaler.fit_transform(df_normal)
X_test = scaler.transform(df.drop(columns=['Fault_Condition']))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

input_dim = X_train.shape[1]

autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

from sklearn.metrics import f1_score, precision_score, recall_score

def find_optimal_threshold(y_true, reconstruction_errors, metric='f1', verbose=True):
    thresholds = np.linspace(min(reconstruction_errors), max(reconstruction_errors), 200)
    best_threshold = None
    best_score = -1

    for t in thresholds:
        preds = (reconstruction_errors > t).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, preds)
        elif metric == 'precision':
            score = precision_score(y_true, preds)
        elif metric == 'recall':
            score = recall_score(y_true, preds)
        else:
            raise ValueError("Invalid metric. Choose from 'f1', 'precision', or 'recall'.")

        if score > best_score:
            best_score = score
            best_threshold = t

    if verbose:
        print(f"✅ Best threshold: {best_threshold:.4f}")
        print(f"⭐ Best {metric} score: {best_score:.4f}")

    return best_threshold

# True labels: 1 = faulty, 0 = normal
y_true = df['Fault_Condition'].apply(lambda x: 1 if x in [1, 2, 3] else 0).values

import numpy as np

X_pred = autoencoder.predict(X_test)
# Calculate per-sensor reconstruction errors
reconstruction_errors_per_sensor = np.square(X_test - X_pred)

# Overall reconstruction error (keep for thresholding)
reconstruction_error = np.mean(reconstruction_errors_per_sensor, axis=1)

# Find best threshold based on F1 score
optimal_threshold = find_optimal_threshold(y_true, reconstruction_error, metric='f1')

# Apply it
autoencoder_preds = (reconstruction_error > optimal_threshold).astype(int)

sensor_names = df.drop(columns=['Fault_Condition']).columns.tolist()

# Save sensor names to a file
import json
with open("iep2_anomalydetection/model/sensor_names.json", "w") as f:
    json.dump(sensor_names, f)

for i, is_anomaly in enumerate(autoencoder_preds):
    if is_anomaly:
        sensor_errors = reconstruction_errors_per_sensor[i]
        # Get indices of top 2 sensors contributing most to anomaly
        top_sensors_idx = np.argsort(sensor_errors)[-2:]
        flagged_sensors = [sensor_names[j] for j in top_sensors_idx]
        print(f"Anomaly detected at index {i} ➜ Likely sensors: {flagged_sensors}")

from sklearn.metrics import classification_report

y_true = df['Fault_Condition'].apply(lambda x: 1 if x in [1,2,3] else 0)
print(classification_report(y_true, autoencoder_preds))

autoencoder.save("iep2_anomalydetection/model/autoencoder_model.h5")

import joblib
joblib.dump(scaler, "iep2_anomalydetection/model/scaler.pkl")

with open("iep2_anomalydetection/model/threshold.txt", "w") as f:
    f.write(str(optimal_threshold))