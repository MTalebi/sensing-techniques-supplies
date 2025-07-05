import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load bridge accelerometer data (14 sensors)
data = load_bridge_data('cable_stay_bridge.csv')
features = extract_features(data, window='5min')

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Define autoencoder architecture
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),  # Bottleneck
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(14, activation='linear')
])

# Train on healthy data (first 30 days)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled[:8640], X_scaled[:8640], 
                epochs=100, validation_split=0.2)

# Anomaly detection on test data
X_pred = autoencoder.predict(X_scaled)
reconstruction_error = np.mean((X_scaled - X_pred)**2, axis=1)
threshold = np.percentile(reconstruction_error[:8640], 99.5)

# Flag anomalies
anomalies = reconstruction_error > threshold 