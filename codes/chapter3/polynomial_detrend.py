# Python implementation
import numpy as np

def polynomial_detrend(x, degree=4):
    """
    Remove polynomial trend from signal
    """
    n = len(x)
    t = np.arange(n)
    
    # Fit polynomial
    coeffs = np.polyfit(t, x, degree)
    trend = np.polyval(coeffs, t)
    
    # Remove trend
    x_clean = x - trend
    
    return x_clean, trend 