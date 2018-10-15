import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time

from collections import namedtuple

import pdb

#########
#Update step constants
########
dT = 1.0
F = np.array([[1.0, dT ],
              [0.0, 1.0]])
B = np.array(np.transpose([np.power(dT, 2) / 2.0, dT]))
Q = np.array([[0.01, 0   ],
              [0,    0.01]])
# Q = np.zeros([2, 2])

#########
#Predict step constants
#########
# H = np.array([[5.6, 0  ],
#               [0,   3.5]])
H = np.identity(2)
R = np.array([[0.5, 0   ],
              [0,   0.005]])
# R = np.zeros([2, 2])

#########
#Helper functions and vars
#########
def updateKinematics(x, u):
    return np.dot(F, x) + np.dot(B, u)
STATE_VARS = ["pos", "vel"]
STATE_INDEX = dict(zip(STATE_VARS, range(len(STATE_VARS))))

class Train:
    def __init__(self, x):
        self.x = x
        print "Truth state: %s" % self.x

    def update(self, u):
        self.x = updateKinematics(self.x, u)
        print "Train moved. New truth state: %s" % self.x


class Sensor:

    def __init__(self, train):
        self.train = train

    def getMeasurement(self):
        #Get "measured" position and velocity (add gaussian noise)
        truth_pos = self.train.x[STATE_INDEX["pos"]]
        measured_pos = np.random.normal(truth_pos, np.sqrt(R[0, 0]))
        truth_vel = self.train.x[STATE_INDEX["vel"]]
        measured_vel = np.random.normal(truth_vel, np.sqrt(R[1, 1]))
        measured_state = [measured_pos, measured_vel]

        #Convert to "volts" using inverse H matrix
        measurement = np.dot(la.inv(H), measured_state)

        return measurement


class KalmanFilter:

    def __init__(self, x, P):
        a = np.dot(P, np.transpose(H))
        b = la.inv(la.multi_dot([H, P, np.transpose(H)]) + R)
        self.K_prime = np.dot(a, b)

        self.x = np.array(x)
        self.P = np.array(P)

    def predict(self, u):
        print "\nRunning predict"
        print "init state: %s" % self.x

        #Predict new state estimate based on world model
        self.x = updateKinematics(self.x, u)

        #Predict new covariance based on world model
        self.P = la.multi_dot([F, self.P, np.transpose(F)]) + Q

        print "final state: %s" % self.x

    def update(self, z):
        print "Running update"
        print "init state: %s, meas: %s" % (self.x, z)

        #Update new state estimate based on sensor model
        self.x = self.x + np.dot(self.K_prime, (z - np.dot(H, self.x)))

        #Update covariance
        self.P = self.P - la.multi_dot([self.K_prime, H, self.P])

        print "final state: %s" % self.x

    def getEstimate(self):
        return self.x, self.P

    def getCovarianceIndex(self, variable_name):
        return self.x._fields.index(variable_name)


class BeliefPlotter:

    x_plot_range = np.arange(-15, 15, 0.001)

    def __init__(self, train, kalman_filter):
        self.train = train
        self.kalman_filter = kalman_filter
        self.initPlot()

    def initPlot(self):
        plt.ion()
        self.fig = plt.figure()
        self.axes = []

        axis = self.fig.add_subplot(211)
        self.axes.append(axis)
        axis = self.fig.add_subplot(212)
        self.axes.append(axis)

        self._initAxisLimits()

    def _initAxisLimits(self):
        axis = self.axes[0]
        axis.set_xlim([-2, 12])
        # axis.set_ylim([0, 1])

        axis = self.axes[1]
        axis.set_xlim([-1, 1])
        # axis.set_ylim([0, 1])

    def clearPlots(self):
        for axis in self.axes:
            axis.cla()

        self._initAxisLimits()

        self.fig.canvas.draw()

    def plotBelief(self):
        for i in range(len(STATE_INDEX)):
            truth_value = self.train.x[i]
            est_value = self.kalman_filter.x[i]
            sigma = np.sqrt(self.kalman_filter.P[i, i])

            axis = self.axes[i]

            axis.plot([truth_value], [0], "y*", markersize=15)
            axis.plot([est_value], [0], "bo")
            est_uncertainty = mlab.normpdf(self.x_plot_range, est_value, sigma)
            axis.plot(self.x_plot_range, est_uncertainty, "b-")

            axis.set_title(STATE_VARS[i])

        self.fig.canvas.draw()

    def plotMeasurement(self, z):
        for i in range(len(STATE_INDEX)):
            axis = self.axes[i]

            meas_val = z[i]
            meas_sigma = np.sqrt(R[i, i])
            meas_y_data = mlab.normpdf(self.x_plot_range, meas_val, meas_sigma)
            axis.plot(self.x_plot_range, meas_y_data, "g-")

            est_value = self.kalman_filter.x[i]
            est_sigma = np.sqrt(self.kalman_filter.P[i, i])
            est_y_data = mlab.normpdf(self.x_plot_range, est_value, est_sigma)

            update_y_data = np.multiply(meas_y_data, est_y_data)
            axis.plot(self.x_plot_range, update_y_data, "r-")
            correct_update_val = self.x_plot_range[np.argmax(update_y_data)]
            axis.axvline(correct_update_val, color="red", linestyle=":")

        self.fig.canvas.draw()


def main():
    init_truth_state = [0.0, .1]
    train = Train(init_truth_state)

    init_estimate = [0.0, .1]
    init_covariance = np.identity(2) * 0.01
    kalman_filter = KalmanFilter(init_estimate, init_covariance)

    plotter = BeliefPlotter(train, kalman_filter)
    sensor = Sensor(train)

    u = 0.01
    for i in range(1000):
        print "=============================="

        ########
        #Make the train move back and forth along the track
        ########
        train_pos = train.x[STATE_INDEX["pos"]]
        #Accelerate forwards
        if train_pos < 2.5:
            u = 0.01
        #Coast
        elif train_pos >= 2.5 and train_pos <= 7.5:
            u = 0
        #Accelerate backwards
        else:
            u = -0.01

        #Update train state and kalman filter estimate
        train.update(u)
        kalman_filter.predict(u)

        #Plot latest state
        plotter.plotBelief()
        raw_input()

        if i % 5 == 0:
            #Get sensor measurement
            z = sensor.getMeasurement()
            #Plot measurement (in green), and where this should cause the
            #kalman filter to update to (in red)
            plotter.plotMeasurement(z)
            raw_input()
            #Update the kalman filter with the measurement
            kalman_filter.update(z)
            #Plot again to watch the update happen
            plotter.plotBelief()
            raw_input()

        plotter.clearPlots()


if __name__ == "__main__":
    main()
