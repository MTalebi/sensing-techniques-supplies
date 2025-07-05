import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Load bridge acceleration data
fs = 200  # Hz
t = np.arange(0, 600, 1/fs)  # 10 min
x = load_bridge_acceleration()

# Welch PSD parameters
nperseg = 4096  # ~20 sec segments
noverlap = 2048  # 50% overlap

# Compute PSD
f, Pxx = signal.welch(x, fs, 
                     window='hann',
                     nperseg=nperseg,
                     noverlap=noverlap)

# Plot results
plt.semilogy(f, np.sqrt(Pxx))
plt.xlabel('Frequency [Hz]')
plt.ylabel('ASD [g/√Hz]')
plt.grid(True) 