import numpy as np

class KalmanFilterCV:
    def __init__(self, dt=0.1):
        # Kalman Filter parameters
        self.dt = dt  # Time step
        self.A = np.array([[1, 0, self.dt, 0], 
                           [0, 1, 0, self.dt], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]])  # State transition matrix
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]])  # Observation matrix
        self.P = np.eye(4)  # Initial covariance matrix
        self.Q = np.eye(4) * 0.01  # Process noise covariance
        self.R = np.eye(2) * 0.1  # Measurement noise covariance
        self.x = np.zeros((4, 1))  # Initial state [x, y, vx, vy]
        self.P = np.eye(4)  # Initial covariance matrix

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Update the state with the measurement
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        # Return the current state
        return self.x.flatten().tolist()

    def reset(self, initial_state):
        self.x = np.zeros((4, 1))
        self.x[:2] = initial_state
        self.P = np.eye(4)