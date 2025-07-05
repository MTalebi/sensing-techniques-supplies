# Modal parameter extraction
def extract_modal_properties(accel, fs):
    """Extract natural frequencies from acceleration data"""
    # Compute power spectral density
    f, Pxx = signal.welch(accel, fs, 
                         nperseg=2048, 
                         noverlap=1024,
                         window='hann')
    
    # Find peaks (natural frequencies)
    peaks, properties = signal.find_peaks(Pxx, 
                                        height=np.max(Pxx)*0.1,
                                        distance=20)
    
    natural_frequencies = f[peaks]
    peak_amplitudes = Pxx[peaks]
    
    return natural_frequencies, peak_amplitudes, f, Pxx

# Usage example
if __name__ == "__main__":
    # Load data
    time, strain, accel, temp, fs = load_sensor_data('bridge_data.csv')
    
    # Temperature compensation
    strain_comp = compensate_strain_temperature(strain, temp)
    
    # Signal conditioning
    accel_clean, strain_clean = condition_signals(accel, strain_comp, fs)
    
    # Modal analysis
    frequencies, amplitudes, f, psd = extract_modal_properties(accel_clean, fs)
    
    # Display results
    print(f"Identified natural frequencies: {frequencies:.2f} Hz")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,2,1)
    plt.plot(time[:1000], accel_clean[:1000])
    plt.title('Filtered Acceleration')
    plt.ylabel('Acceleration (g)')
    
    plt.subplot(2,2,2)
    plt.semilogy(f, psd)
    plt.plot(frequencies, amplitudes, 'ro', markersize=8)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (g²/Hz)')
    
    plt.tight_layout()
    plt.show() 