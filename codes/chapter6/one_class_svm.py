from sklearn.svm import OneClassSVM
from sklearn.covariance import MinCovDet

# Robust covariance estimation
robust_cov = MinCovDet(random_state=42)
robust_cov.fit(Z_baseline)

# One-class SVM with RBF kernel
svm_detector = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
svm_detector.fit(Z_baseline)

# Anomaly detection
robust_distances = robust_cov.mahalanobis(Z_test)
svm_scores = svm_detector.decision_function(Z_test) 