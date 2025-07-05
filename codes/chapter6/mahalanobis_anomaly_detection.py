import numpy as np
from scipy import stats
from sklearn.covariance import EmpiricalCovariance

# Step 1: Estimate baseline statistics
baseline_cov = EmpiricalCovariance()
baseline_cov.fit(Z_baseline)  # Z_baseline: N x d feature matrix
mu = Z_baseline.mean(axis=0)
Sigma_inv = baseline_cov.precision_

# Step 2: Compute Mahalanobis distance for new data
def mahalanobis_distance(z_new):
    diff = z_new - mu
    return np.sqrt(diff @ Sigma_inv @ diff.T)

# Step 3: Set threshold for 1% false alarm rate
d = Z_baseline.shape[1]  # number of features
threshold = stats.chi2.ppf(0.99, d)

# Step 4: Detect anomalies
distances = [mahalanobis_distance(z) for z in Z_test]
anomalies = np.array(distances) > threshold 