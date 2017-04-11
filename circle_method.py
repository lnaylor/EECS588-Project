#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from generate_data import Dataset
from defense import mesh
from defense2 import BetterThanFrank
from qclp_cplex import QCLP

class Anomaly_Detector:
    
    def __init__(self, method, data, r, window=200, grid_width=0, time_window=50, percent=.25, NN=10, NNdefense_on = True):
        self.__method = method
        self.__data = np.array(data)[:window]
        self.__r = r
        self.__center = np.mean(data, axis=0)
        self.__saved_center = np.mean(data, axis=0)
        self.__grid_centers = []
        self.__grid_width = grid_width
        self.__window = window
        self.__time_window = time_window
        self.__counter = 0
        self.__percent = percent
        self.__NN = NN
        self.__NNdefense = NNdefense_on

        if method == 'nearest-out':
            from qclp_cplex import QCLP

    def get_r(self):
        return self.__r
    def get_data(self):
        return self.__data
    def get_grid_width(self):
        return self.__grid_width
    def set_grid_width(self, w):
        self.__grid_width = w
    def get_time_window(self):
        return self.__time_window

    def add_grid_center(self, c):
        self.__grid_centers.append(c)
    def get_grid_centers(self):
        return self.__grid_centers

    def add_point(self, point):
        if self.__NNdefense:
            if self.__counter >= self.__time_window:
                print self.__center
                buff_data = self.__data[-100:]
                past_points = BetterThanFrank(buff_data)
                suspicious_pts = past_points.iGEM(NN=self.__NN, num_suspect = int(self.__percent*self.__time_window))
                new_buff_data = np.array([buff_data[i] for i in range(len(buff_data)) if i not in suspicious_pts])
                self.__data = np.append(self.__data[:-100], new_buff_data, axis=0)

#                plt.figure()
#                plt.scatter(buff_data[:, 0], buff_data[:, 1], color='b', s=3)
#                for i in range(len(suspicious_pts)):
#                    plt.scatter(buff_data[suspicious_pts[i], 0], buff_data[suspicious_pts[i], 1], color='k', s=5)
##                plt.show()
#                plt.savefig('defense.png')
#                plt.close()

                self.__center = np.mean(self.__data, axis=0)
                print self.__center
                print
                self.__counter = 0
            
        if len(self.__data) < self.__window:
            self.__data = np.append(self.__data, point, axis=0)
            self.__center = np.mean(self.__data, axis=0)

        elif self.__method=="random-out":
            random_i = random.randint(0, len(self.__data)-1)
            self.__data = np.delete(self.__data, random_i, 0)
            self.__data = np.append(self.__data, point, 0)
            self.__center = np.mean(self.__data, axis=0)

        elif self.__method=='nearest-out':
            distances = np.linalg.norm(self.__data - point, axis=1)
            i = np.argmin(distances)
            self.__data = np.delete(self.__data, i, 0)
            self.__data = np.append(self.__data, point, 0)
            self.__center = np.mean(self.__data, axis=0)
        
        elif self.__method=='average-out':
            new_center = self.__center + ((1.0/len(self.__data))*(point-self.__center))
            self.__center = new_center[0]

        else:
            print("invalid removal method")
         
        self.__counter += 1

    def classify_point(self, point):

        for c in self.__grid_centers:
            xMin = c[0] - self.__grid_width
            xMax = c[0] + self.__grid_width
            yMin = c[1] - self.__grid_width
            yMax = c[1] + self.__grid_width
            if point[0] <= xMax and point[0] >= xMin and point[1] <= yMax and point[1] >= yMin:
                return False

        distance = np.linalg.norm(point - self.__center)
        if distance < self.__r:
            return True
        return False

    def get_center(self):
        return self.__center


def simple_attack(data, target, r):
    center = np.mean(data, axis=0)
    direction = target - center
    new_point = center + (r*direction)/np.linalg.norm(direction)
    return new_point

def greedy_optimal_attack(data, target, r):
    center = np.mean(data, axis=0)
    direction = target - center
    new_point = QCLP(data, direction, r)
    return new_point

def main():

    c = 0
#####Plotting

#    center = detector.get_center()
#    fig = plt.figure()
#    plt.scatter(data.X[data.Y == True, 0], data.X[data.Y == True, 1], color='r', s=1)
#    plt.scatter(data.X[data.Y == False, 0], data.X[data.Y == False, 1], color='b', s=1)
#    ax = fig.add_subplot(1,1,1)
#    circ = plt.Circle(center, 2.0, fill = False)
#    ax.add_patch(circ)
#    plt.show()
#


if __name__ == '__main__':
    main()
