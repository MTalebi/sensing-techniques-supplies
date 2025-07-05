import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Load bridge acceleration data
x = np.load("bridge_accel.npy")  # 1 hour @ 200 Hz

# Fit AR(6) model
ar_model = AutoReg(x, lags=6, old_names=False).fit()

# Extract coefficients
a_coeffs = ar_model.params[1:]  # AR coefficients
sigma2 = ar_model.sigma2        # Noise variance

# One-step prediction
x_pred = ar_model.predict(start=len(x)-100, 
                         end=len(x)-1)

print(f"AR coefficients: {a_coeffs}")
print(f"Prediction RMSE: {np.sqrt(np.mean((x[-100:] - x_pred)**2))}") 