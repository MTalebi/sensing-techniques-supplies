import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA

# Load bridge acceleration data
data = pd.read_csv('bridge_accel.csv')
x = data['acceleration'].values
fs = 200  # Hz

# 1. Preprocessing
# Detrend
x_detrend = signal.detrend(x)

# Anti-alias filter
b, a = signal.butter(4, 20, fs=fs)
x_clean = signal.filtfilt(b, a, x_detrend)

# 2. Feature extraction
def extract_features(signal, window_size=1000):
    features = []
    for i in range(0, len(signal)-window_size, window_size//2):
        segment = signal[i:i+window_size]
        
        # Time domain features
        rms = np.sqrt(np.mean(segment**2))
        crest = np.max(np.abs(segment)) / rms
        skewness = np.mean((segment - np.mean(segment))**3) / np.std(segment)**3
        
        # Frequency domain features
        freqs, psd = signal.welch(segment, fs=fs)
        peak_freq = freqs[np.argmax(psd)]
        
        features.append([rms, crest, skewness, peak_freq])
    
    return np.array(features)

# Extract features
features = extract_features(x_clean)

# 3. Dimensionality reduction
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(features)
#hello no
print(f"Original: {features.shape[1]} features")
print(f"Reduced: {reduced_features.shape[1]} features")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}") 
