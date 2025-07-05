from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

# State-space model: [position, velocity]
A = np.array([[1, dt], [0, 1]])    # dt = sampling interval
C = np.array([[0, 1]])             # measure acceleration
B = np.array([[dt**2/2], [dt]])    # input matrix

# Noise covariances
Q = 1e-6 * np.eye(2)               # process noise
R = np.array([[1e-4]])             # measurement noise

# Initialize Kalman filter
x_hat = np.zeros((2, len(y_accel)))
P = np.eye(2)

for k in range(1, len(y_accel)):
    # Predict
    x_pred = A @ x_hat[:, k-1]
    P_pred = A @ P @ A.T + Q
    
    # Update
    K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)
    x_hat[:, k] = x_pred + K @ (y_accel[k] - C @ x_pred)
    P = (np.eye(2) - K @ C) @ P_pred

# Extract virtual displacement
displacement = x_hat[0, :] 