#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from generate_data import Dataset
from defense import mesh
#from qclp_cplex import QCLP

class Anomaly_Detector:
    
    def __init__(self, method, data, r, window=200, grid_width=0):
        self.__method = method
        self.__data = np.array(data)
        self.__r = r
        self.__center = np.mean(data, axis=0)
        self.__grid_centers = []
        self.__grid_width = grid_width
        self.__window = window

    def get_r(self):
        return self.__r
    def get_data(self):
        return self.__data
    def get_grid_width(self):
        return self.__grid_width
    def set_grid_width(self, w):
        self.__grid_width = w

    def add_grid_center(self, c):
        self.__grid_centers.append(c)
    def get_grid_centers(self):
        return self.__grid_centers

    def add_point(self, point):
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
            self.__center = self.__center + ((1.0/len(self.__data))*(point-self.__center))

        else:
            print("invalid removal method")
    

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

    method = 'random-out'
#    method = 'nearest-out'
    target = [2, 2]
    
#    # data = np.random.normal(size=(50, 2))
    data = Dataset(p=2, n=1000, phi=0.05)
    data.generate_data(standard=True)
    training = data.X
    training = np.reshape(training, (len(training), 2))
    (min_x, min_y) = np.min(training, axis=0)
    (max_x, max_y) = np.max(training, axis=0)
    min_x -= 10
    min_y -= 10
    max_x += 10
    max_y += 10
    xFine = 10
    yFine = 10
    thresh = 0.5

    m = mesh(min_x, max_x, min_y, max_y, xFine, yFine)
    detector = Anomaly_Detector(method, training[:2], 2.0)
    for t in training[2:]:
        old_center = detector.get_center()
        detector.add_point(np.reshape(t, (1, len(t))))
        new_center = detector.get_center()
        m.addLine(old_center, new_center)
        print new_center

    
    detector.set_grid_width(m.width)
    baseline = m.percentage()
    m.mode= 'test'
    
    normal_pts = []
#    while not detector.classify_point(target):
    for i in range(1000):
        old_center = detector.get_center()
        if random.random() < .25:
            if method == 'random-out' or method == 'average-out':
                new_point = simple_attack(detector.get_data(), target, detector.get_r())
            else:
                new_point  = greedy_optimal_attack(detector.get_data(), target, detector.get_r())
        else:
            pt = data.generate_new_points(1)
            (true_value, new_point) = (pt[0], pt[1][0])
            normal_pts.append((new_point, true_value))

        new_point = np.reshape(new_point, (1, len(new_point)))
        detector.add_point(new_point)
        new_center = detector.get_center()
        m.addLine(old_center, new_center)
        c, w = m.checkPoints(baseline, thresh)
        for cen in c:
            detector.add_grid_center(c)
        print detector.classify_point(target)

    normal_pts = np.array(normal_pts)
    true_positive = 0
    for n in normal_pts:
        if detector.classify_point(n[0]) != n[1]:
            true_positive += 1
    print 'true positive: ', float(true_positive)/float(len(normal_pts))

#
#    center = detector.get_center()
#    fig = plt.figure()
#    plt.scatter(data.X[data.Y == True, 0], data.X[data.Y == True, 1], color='r', s=1)
#    plt.scatter(data.X[data.Y == False, 0], data.X[data.Y == False, 1], color='b', s=1)
#    ax = fig.add_subplot(1,1,1)
#    circ = plt.Circle(center, 2.0, fill = False)
#    ax.add_patch(circ)
#    plt.show()
#    print(center)
#    print(detector.classify_point([1, 1]))
#
#    while (not detector.classify_point([1,1])):
#
#        detector.add_point([simple_attack(data, [1, 1], 1.0)])
#        print(detector.get_center())
#
#    print(detector.classify_point([1,1]))


if __name__ == '__main__':
    main()
