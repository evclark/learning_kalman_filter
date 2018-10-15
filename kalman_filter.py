import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time

from collections import namedtuple

import pdb

#Update step constants
dT = 1.0
F = np.array([[1.0, dT ],
              [0.0, 1.0]])
B = np.array(np.transpose([(dT ** 2) / 2.0, dT]))
def updateKinematics(x, u):
    new_x = np.dot(F, x) + np.dot(B, u)
    new_x = State(*new_x)
    return new_x
Q = np.array([[0.01, 0   ],
              [0,    0.01]])

#Predict step constants
H = np.array([[5.6, 0  ],
              [0,   3.5]])
R = np.array([[0.5, 0   ],
              [0,   0.01]])

#Create State class to keep track of state
State = namedtuple("State", ["pos", "vel"])
#Make fields available through dictionary-style access
State.getVal = lambda self, key: self.__dict__[key]

#Create StateVarPlot class to plot a state variable
StateVarPlot = namedtuple("StateVarPlot", ["axis", 'truth', 'estimate',
                                        'uncertainty', 'measurement', 'update'])

class Train:
    def __init__(self, x):
        self.x = x

    def update(self, u):
        self.x = updateKinematics(self.x, u)
        print self.x


class Sensor:

    def __init__(self, train):
        self.train = train

    def getMeasurement(self):
        #Get "measured" position and velocity (add gaussian noise)
        measured_pos = np.random.normal(self.train.x.pos, R[0, 0])
        measured_vel = np.random.normal(self.train.x.vel, R[1, 1])

        #Convert to "volts" using inverse H matrix
        measured_state = State(measured_pos, measured_vel)
        measured_state = np.dot(la.inv(H), measured_state)

        return State(measured_pos, measured_vel)


class KalmanFilter:

    def __init__(self, x, P):
        a = np.dot(P, np.transpose(H))
        b = la.inv(la.multi_dot([H, P, np.transpose(H)]) + R)
        self.K_prime = np.dot(a, b)

        self.x = x
        self.P = P

    def predict(self, u):
        #Predict new state estimate based on world model
        self.x = updateKinematics(self.x, u)

        #Predict new covariance based on world model
        self.P = la.multi_dot([F, self.P, np.transpose(F)]) + Q

    def update(self, z):
        #Update new state estimate based on sensor model
        new_x = self.x + np.dot(self.K_prime, (z - np.dot(H, self.x)))
        self.x = State(*new_x)

        #Pre
        self.P = self.P - la.multi_dot([self.K_prime, H, self.P])


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
        ax.set_xlim([-2, 12])
        ax.set_ylim([0, 1])
        truth, = ax.plot([0], [0], "y*", markersize=15, alpha=0.5)
        estimate, = ax.plot([0], [0], "bo", alpha=0.5)
        uncertainty, = ax.plot([0], [0])
        measurement, = ax.plot([0], [0])
        update, = ax.plot([0], [0])
        self.plots["pos"] = StateVarPlot(ax, truth, estimate, uncertainty,
                                                             measurement, update)

        ax = self.fig.add_subplot(212)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 1])
        truth, = ax.plot([0], [0], "y*", markersize=15, alpha=0.5)
        estimate, = ax.plot([0], [0], "bo", alpha=0.5)
        uncertainty, = ax.plot([0], [0])
        measurement, = ax.plot([0], [0])
        update, = ax.plot([0], [0])
        self.plots["vel"] = StateVarPlot(ax, truth, estimate, uncertainty,
                                                            measurement, update)

    def plotBelief(self):
        #Update data in plot Line objects
        self.plotStateVariable("pos")
        self.plotStateVariable("vel")

        #Redraw the plots with the new data
        self.fig.canvas.draw()



    def plotMeasurement(self, z):
        for var_name in ["pos", "vel"]:
            measurement_plot =  self.plots[var_name].measurement
            update_plot = self.plots[var_name].update

            est_value = self.kalman_filter.x.getVal(var_name)
            #Get position uncertainty
            cov_index = self.kalman_filter.getCovarianceIndex(var_name)
            est_variance = self.kalman_filter.P[cov_index, cov_index]

            x = np.arange(-15, 15, 0.001)
            y = mlab.normpdf(x, est_value, est_variance)

            meas_val = z.getVal(var_name)
            meas_cov = R[cov_index, cov_index]
            y2 = mlab.normpdf(x, meas_val, meas_cov)
            measurement_plot.set_xdata(x)
            measurement_plot.set_ydata(y2)

            y3 = np.multiply(y, y2)
            # y3 /= np.max(y3)
            update_plot.set_xdata(x)
            update_plot.set_ydata(y3)

        self.fig.canvas.draw()

    def clearMeasurement(self):
        for var_name in ["pos", "vel"]:
            measurement_plot =  self.plots[var_name].measurement
            update_plot = self.plots[var_name].update

            measurement_plot.set_xdata([0])
            measurement_plot.set_ydata([0])
            update_plot.set_xdata([0])
            update_plot.set_ydata([0])

        self.fig.canvas.draw()

    def plotStateVariable(self, var_name):
        #Get true positions
        truth_value = self.train.x.getVal(var_name)
        #Get estimated position
        est_value = self.kalman_filter.x.getVal(var_name)
        #Get position uncertainty
        cov_index = self.kalman_filter.getCovarianceIndex(var_name)
        est_variance = self.kalman_filter.P[cov_index, cov_index]

        print "%s variance: %s" % (var_name, est_variance)

        #Get plot Line objects for this state variable
        truth_plot = self.plots[var_name].truth
        est_plot =  self.plots[var_name].estimate
        uncertainty_plot = self.plots[var_name].uncertainty

        #Update them with the latest data
        truth_plot.set_xdata([truth_value])
        est_plot.set_xdata([est_value])
        x = np.arange(-15, 15, 0.001)
        y = mlab.normpdf(x, est_value, est_variance)
        uncertainty_plot.set_xdata(x)
        uncertainty_plot.set_ydata(y)



def main():
    init_truth_state = State(0.1, .11)
    train = Train(init_truth_state)

    init_estimate = State(0.0, .1)
    init_covariance = np.identity(2) * 0.01
    kalman_filter = KalmanFilter(init_estimate, init_covariance)

    plotter = BeliefPlotter(train, kalman_filter)
    sensor = Sensor(train)

    u = 0.01
    for i in range(1000):
        if train.x.pos < 2.5:
            u = 0.01
        elif train.x.pos >= 2.5 and train.x.pos <= 7.5:
            u = 0
        else:
            u = -0.01

        #Plot latest state
        plotter.plotBelief()
        raw_input()

        #Update train state and kalman filter estimate
        train.update(u)
        #Give the kalman filter a measurement of the train every 100 iterations
        if i % 5 == 0:
            z = sensor.getMeasurement()
            plotter.plotMeasurement(z)
            raw_input()
            kalman_filter.update(z)
        else:
            kalman_filter.predict(u)
            plotter.clearMeasurement()

        # Control animation speed







if __name__ == "__main__":
    main()
