import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter


class UKF:
    def __init__(self):
        self.current_time = 0.0

        dt = 0.1  # standard dt

        # Create sigma points
        self.points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2.0, kappa=-1)

        self.kf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, fx=self.fx, hx=self.hx, points=self.points)

        # Initial state: [x, vx, y, vy, z, vz]
        self.kf.x = np.array([1., 0, 300., 0, 1., 0])

        # Initial uncertainty
        self.kf.P *= 1

        # Measurement noise covariance matrix
        z_std = 0.2
        self.kf.R = np.diag([z_std ** 2, z_std ** 2, z_std ** 2])

        # Process noise covariance matrix
        process_noise_stds = [3.0 ** 2, 0.1 ** 2, 3 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2]

        self.kf.Q = np.diag(process_noise_stds)

    def fx(self, x, dt):
        """ State transition function """
        F = np.array([[1, dt, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]], dtype=float)
        return np.dot(F, x)

    def hx(self, x):
        """ Measurement function [x, y, z] """
        return np.array([x[0], x[2], x[4]])

    def predict(self, time):
        """ Predict the next state """
        delta_t = time - self.current_time
        self.kf.predict(dt=delta_t)
        self.current_time = time

    def update(self, time, z):
        """ Update filter with measurements z. """
        self.predict(time)
        self.kf.update(z)
