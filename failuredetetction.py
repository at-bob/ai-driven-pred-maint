import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Set working directory to where this script is located
os.chdir(os.path.dirname(__file__))

# Create model folder if not exists
os.makedirs("iep1_failuredetection/model", exist_ok=True)

# ===============================
# 1. Load & Preprocess Data
# ===============================
df = pd.read_csv('engine_failure_dataset.csv', parse_dates=['Time_Stamp'])
df.set_index('Time_Stamp', inplace=True)

# Map Operational Mode
mode_map = {'Idle': 1, 'Cruising': 2, 'Heavy Load': 3}
df['Operational_Mode'] = df['Operational_Mode'].map(mode_map)

# Create binary target
df['Risk_Flag'] = df['Fault_Condition'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# Features & Target
X = df.drop(columns=['Fault_Condition', 'Risk_Flag'])
y = df['Risk_Flag']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===============================
# 2. Build Neural Network
# ===============================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


from sklearn.utils import class_weight

# Compute class weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
# ===============================
# 3. Train Model
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# 4. Evaluate Model
# ===============================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 5. Save Model & Scaler
# ===============================
model.save("iep1_failuredetection/model/failure_classifier_binary.h5")
joblib.dump(scaler, "iep1_failuredetection/model/scaler.pkl")

print("\nModel and scaler saved successfully.")