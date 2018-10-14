import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time

from collections import namedtuple

import pdb

dT = 1.0
F = np.array([[1.0, dT ],
              [0.0, 1.0]])
B = np.array(np.transpose([(dT ** 2) / 2.0, dT]))
def updateKinematics(x, u):
    new_x = np.dot(F, x) + np.dot(B, u)
    new_x = State(*new_x)
    return new_x

#Create State class to keep track of state
State = namedtuple("State", ["pos", "vel"])
#Make fields available through dictionary-style access
State.getVal = lambda self, key: self.__dict__[key]

#Create StateVarPlot class to plot a state variable
StateVarPlot = namedtuple("StateVarPlot", ["axis", 'truth', 'estimate', 'uncertainty'])

class Train:

    POS_MEASUREMENT_SIGMA = 0.5
    VEL_MEASUREMENT_SIGMA = 0.1

    def __init__(self, x):
        self.x = x

    def update(self, u):
        self.x = updateKinematics(self.x, u)
        print self.x


class Sensor:

    def __init__(self, train):
        self.train = train

    def getMeasurement(self):
        #Multiply by 5.6 to convert to "volts" so we can have an interesting
        #H matrix
        measured_pos = 5.6 * np.random.normal(self.train.x.pos,
                                                self.POS_MEASUREMENT_SIGMA)
        #Multiply by 3.2 to convert to "volts" so we can have an interesting
        #H matrix
        measured_vel = 3.2 * np.random.normal(self.train.x.vel,
                                                self.VEL_MEASUREMENT_SIGMA)

        return State(measured_pos, measured_vel)


class KalmanFilter:

    def __init__(self, x, P):
        self.Q = np.zeros([2, 2])

        self.x = x
        self.P = P

    def predict(self, u):
        #Calculate new state estimate
        new_x = updateKinematics(self.x, u)
        self.x = new_x

        #Calculate new covariance estimate
        new_P = np.linalg.multi_dot(F, self.P, np.transpose(F)) + self.Q
        self.P = new_P

    def update(self):
        pass

    def getEstimate(self):
        return self.x, self.P

    def getCovarianceIndex(self, variable_name):
        return self.x._fields.index(variable_name)


class BeliefPlotter:

    def __init__(self, train, kalman_filter):
        self.train = train
        self.kalman_filter = kalman_filter
        self.initPlot()

    def initPlot(self):
        plt.ion()
        self.fig = plt.figure()

        self.plots = {}

        ax = self.fig.add_subplot(211)
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 1])
        truth, = ax.plot([0], [0], "y*", markersize=15, alpha=0.5)
        estimate, = ax.plot([0], [0], "bo", alpha=0.5)
        x_axis = np.arange(0, 10, 0.001)
        uncertainty, = ax.plot(x_axis, [0] * len(x_axis))
        self.plots["pos"] = StateVarPlot(ax, truth, estimate, uncertainty)

        ax = self.fig.add_subplot(212)
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 1])
        truth, = ax.plot([0], [0], "y*", markersize=15, alpha=0.5)
        estimate, = ax.plot([0], [0], "bo", alpha=0.5)
        x_axis = np.arange(0, 10, 0.001)
        uncertainty, = ax.plot(x_axis, [0] * len(x_axis))
        self.plots["vel"] = StateVarPlot(ax, truth, estimate, uncertainty)



    def plotBelief(self):
        self.plotStateVariable("pos")
        self.plotStateVariable("vel")

        self.fig.canvas.draw()

    def plotStateVariable(self, var_name):


        #Get true positions
        truth_value = self.train.x.getVal(var_name)
        #Get estimated position
        est_value = self.kalman_filter.x.getVal(var_name)
        #Get position uncertainty
        cov_index = self.kalman_filter.getCovarianceIndex(var_name)
        est_variance = self.kalman_filter.P[cov_index, cov_index]

        print truth_value, est_value

        truth_plot = self.plots[var_name].truth
        est_plot =  self.plots[var_name].estimate
        uncertainty_plot = self.plots[var_name].uncertainty

        truth_plot.set_xdata([truth_value])
        est_plot.set_xdata([est_value])
        x_axis = np.arange(0, 10, 0.001)
        uncertainty_plot.set_ydata(mlab.normpdf(x_axis, est_value, est_variance))


def main():
    init_truth_state = State(0.1, .11)
    train = Train(init_truth_state)

    init_estimate = State(0.0, .1)
    init_covariance = np.identity(2) * 0.1
    kalman_filter = KalmanFilter(init_estimate, init_covariance)

    plotter = BeliefPlotter(train, kalman_filter)

    for i in range(500):
        plotter.plotBelief()

        train.update(0.01)
        time.sleep(0.1)






if __name__ == "__main__":
    main()
