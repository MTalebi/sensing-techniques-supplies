import numpy as np, pandas as pd
from scipy import signal

def process_daily_data(csv_file):
    """Complete time-domain processing pipeline"""
    # Load data
    df = pd.read_csv(csv_file)
    x = df['acc'].values
    fs = 200.0
    
    # 1. Detrend
    p = np.polyfit(np.arange(len(x)), x, 4)
    trend = np.polyval(p, np.arange(len(x)))
    x -= trend
    
    # 2. Hampel filter
    x = hampel_filter(x, k=10)
    
    # 3. Anti-alias and decimate
    b = signal.firwin(91, 20.0/(0.5*fs))
    x = signal.lfilter(b, 1.0, x)[90:]
    x = signal.decimate(x, 4)
    fs = 50.0
    
    # 4. Windowed statistics
    features = extract_features(x, fs, window_min=5)
    
    return features

def extract_features(x, fs, window_min=5):
    """Extract statistical features in sliding windows"""
    win_samples = int(window_min * 60 * fs)
    features = []
    
    for i in range(0, len(x)-win_samples, win_samples):
        seg = x[i:i+win_samples]
        
        # Compute features
        rms = np.sqrt(np.mean(seg**2))
        crest = np.max(np.abs(seg)) / rms
        skew = scipy.stats.skew(seg)
        kurt = scipy.stats.kurtosis(seg)
        
        features.append([rms, crest, skew, kurt])
    
    return np.array(features) 