import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Load sensor data
def load_sensor_data(filename):
    """Load multi-channel sensor data from CSV"""
    df = pd.read_csv(filename)
    fs = 200  # Sample rate in Hz
    
    # Extract time and sensor channels
    time = df['time'].values
    strain = df['strain_ch1'].values  # microstrains
    accel = df['accel_ch1'].values    # g units
    temp = df['temperature'].values   # Celsius
    
    return time, strain, accel, temp, fs

# Temperature compensation for strain
def compensate_strain_temperature(strain, temp):
    """Remove temperature effects from strain data"""
    # Linear compensation model
    # Typical steel: 12 µε/°C thermal expansion
    alpha_steel = 12e-6  # /°C
    temp_ref = np.mean(temp)
    
    # Remove thermal strain
    thermal_strain = alpha_steel * (temp - temp_ref) * 1e6  # Convert to µε
    compensated_strain = strain - thermal_strain
    
    return compensated_strain

# Signal conditioning
def condition_signals(accel, strain, fs):
    """Apply filtering and conditioning to sensor signals"""
    # Design anti-aliasing filter
    nyquist = fs / 2
    cutoff = 25  # Hz
    b, a = signal.butter(4, cutoff/nyquist, 'low')
    
    # Filter signals
    accel_filt = signal.filtfilt(b, a, accel)
    strain_filt = signal.filtfilt(b, a, strain)
    
    return accel_filt, strain_filt 