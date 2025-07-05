import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class HybridSHMDetector:
    def __init__(self, n_features=38, latent_dim=4, n_clusters=4):
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
        self.kmeans = None
        self.threshold = None
        
    def build_autoencoder(self):
        """Build autoencoder architecture"""
        input_layer = tf.keras.layers.Input(shape=(self.n_features,))
        
        # Encoder
        encoded = tf.keras.layers.Dense(16, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(self.latent_dim, 
                                      activation='relu')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(16, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(self.n_features, 
                                      activation='linear')(decoded)
        
        self.autoencoder = tf.keras.Model(input_layer, decoded)
        self.encoder = tf.keras.Model(input_layer, encoded)
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
    def train(self, healthy_data):
        """Train on healthy bridge data"""
        # Normalize data
        X_scaled = self.scaler.fit_transform(healthy_data)
        
        # Train autoencoder
        self.build_autoencoder()
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Get latent representations
        latent_codes = self.encoder.predict(X_scaled)
        
        # Train k-means on latent space
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                           random_state=42)
        cluster_labels = self.kmeans.fit_predict(latent_codes)
        
        # Set anomaly threshold (99.5th percentile)
        X_pred = self.autoencoder.predict(X_scaled)
        reconstruction_errors = np.mean((X_scaled - X_pred)**2, axis=1)
        self.threshold = np.percentile(reconstruction_errors, 99.5)
        
        return history
    
    def detect_anomaly(self, new_data):
        """Detect anomalies in new data"""
        X_scaled = self.scaler.transform(new_data.reshape(1, -1))
        
        # Reconstruction error check
        X_pred = self.autoencoder.predict(X_scaled)
        recon_error = np.mean((X_scaled - X_pred)**2)
        
        # Cluster assignment check
        latent_code = self.encoder.predict(X_scaled)
        cluster = self.kmeans.predict(latent_code)[0]
        cluster_distance = np.min(
            np.linalg.norm(latent_code - self.kmeans.cluster_centers_, 
                          axis=1)
        )
        
        # Combined decision
        is_anomaly = (recon_error > self.threshold) or \
                    (cluster_distance > np.std(
                        np.linalg.norm(
                            self.encoder.predict(
                                self.scaler.transform(healthy_data)
                            ) - self.kmeans.cluster_centers_[cluster]
                        )
                    ) * 2.5)
        
        return {
            'anomaly': is_anomaly,
            'recon_error': recon_error,
            'cluster': cluster,
            'cluster_distance': cluster_distance
        }

# Usage example
detector = HybridSHMDetector()
detector.train(healthy_bridge_data)

# Real-time monitoring loop
for new_window in streaming_data:
    result = detector.detect_anomaly(new_window)
    if result['anomaly']:
        trigger_alert(result) 