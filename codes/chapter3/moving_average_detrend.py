# For strain signals
def moving_average_detrend(x, window_size):
    """
    Remove moving average trend
    """
    # Choose window >> 1/f_mode
    # For 1Hz mode: window ~ 50-100 samples
    
    trend = np.convolve(x, 
                       np.ones(window_size)/window_size, 
                       mode='same')
    
    x_clean = x - trend
    
    return x_clean, trend 