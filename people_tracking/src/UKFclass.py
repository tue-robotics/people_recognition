import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common.discretization import Q_discrete_white_noise
# from numpy.random import randn
#
# import random
# import matplotlib.pyplot as plt
import scipy

# def sqrt_func(x):
#     """ Sqrt functions that prevents covariance matrix P to become slowly not symmetric or positive definite"""
#     try:
#         result = scipy.linalg.cholesky(x)
#     except scipy.linalg.LinAlgError:
#         x = (x + x.T)/2
#         result = scipy.linalg.cholesky(x)
#     return result


class UKF:

    def __init__(self):
        self.current_time = 0

        dt = 0.1  # standard dt

        # Create sigma points
        self.points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)#, sqrt_method=sqrt_func)

        self.kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=self.fx, hx=self.hx, points=self.points)

        self.kf.x = np.array([1., 0, 1., 0])  # Initial state
        self.kf.P *= 1  # Initial uncertainty

        z_std = 0.2
        self.kf.R = np.diag([z_std ** 2, z_std ** 2])  # Measurement noise covariance matrix

        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var= 3 ** 2, block_size=2) #

    #  https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html
    def fx(self, x, dt):
        """ Function that returns the state x transformed by the state transition function. (cv model)
            Assumption:
            *   x = [x, vx, z, vz]^T
        """
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]], dtype=float)
        return np.dot(F, x)

    def hx(self, x):
        """Measurement function - convert the state into a measurement where measurements are [x_pos, z_pos] """
        return np.array([x[0], x[2]])

    def predict(self, time):
        """ update the kalman filter by predicting value."""
        delta_t = time - self.current_time
        self.kf.predict(dt=delta_t)
        self.current_time = time

    def update(self, time, z):
        """  Update filter with measurements z."""
        self.predict(time)
        self.kf.update(z)


# # Example with varying dt
# z_std = 0.1
# zs = [[i + randn() * z_std, i + randn() * z_std] for i in range(50)]  # Measurements
#
# start = 0
# step = 0.1
# end = 10
# times = [start + step * i for i in range(int((end - start) / step) + 1)]
# times = [x + random.uniform(0, 0.05) for x in times]
#
# print(times)
# xs = []
# ys = []
# xk = []
# yk = []
#
# i = -1
# time_last = 0
#
# ukf = UKF()
#
# for z in zs:
#     i = i + 1
#     # delta_t = times[i] - time_last
#     # print(times[i], delta_t)
#     ukf.predict(times[i])
#     ukf.update(times[i], z)
#     # time_last = times[i]
#     # kf.update(z)
#     # print(kf.x, 'log-likelihood', kf.log_likelihood)
#     xs.append(z[0])
#     ys.append(z[1])
#     xk.append(ukf.kf.x[0])
#     yk.append(ukf.kf.x[2])
# # print(xk)
#
# fig, ax = plt.subplots()
# # measured = [list(t) for t in zs]
# ax.scatter(xs, ys, s=5)
# ax.plot(xk, yk)
#
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Z-axis')
#
# # Display the plot
# plt.show()
