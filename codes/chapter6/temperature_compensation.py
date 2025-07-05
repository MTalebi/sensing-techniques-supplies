import numpy as np
from sklearn.linear_model import LinearRegression

# Fit temperature model on baseline data
temp_model = LinearRegression()
temp_model.fit(T_baseline.reshape(-1,1), z_baseline)

# Remove temperature effects from new data
z_corrected = z_new - temp_model.predict(T_new.reshape(-1,1))

# Check residual correlation
correlation = np.corrcoef(z_corrected, T_new)[0,1]
print(f"Residual correlation: {correlation:.3f}") 