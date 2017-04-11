#!/usr/bin/env python

import numpy as np
import random
import matplotlib as plt
from generate_data import Dataset
from defense import mesh
from circle_method import *


def main():
    num_trials = 1    
    method = 'random-out'
#    method = 'nearest-out'

#    defense_method = 'grid'
    defense_method = 'NN'
#    defense_method = 'none'
    
    radius = 2
    target = [1.7, 1.7]
    attacker_percentage = .25
    
    data_window = 200
    buffer_window = 50
    num_neighbors = 10

    avg_iterations = []
    avg_true_positive = []
    for k in range(num_trials):
        data = Dataset(p=radius, n=1000, phi=0.05)
        data.generate_data(standard=True)
         
        (min_x, min_y) = np.min(data.X, axis=0)
        (max_x, max_y) = np.max(data.X, axis=0)
        min_x -= 10
        min_y -= 10
        max_x += 10
        max_y += 10
        xFine = 100
        yFine = 100
        thresh = 0.01

        m = mesh(min_x, max_x, min_y, max_y, xFine, yFine)
        
        if method == 'nearest-out':
            from qclp_cplex import QCLP

        if defense_method == 'grid':
            detector = Anomaly_Detector(method, data.X[:1], radius, data_window, NNdefense_on = False)
            for t in data.X[1:]:
                old_center = detector.get_center()
                detector.add_point(np.reshape(t, (1, len(t))))
                new_center = detector.get_center()
                m.addLine(old_center, new_center)
                
            detector.set_grid_width(m.width)
            baseline = m.percentage()
            m.mode = 'test'
        
        if defense_method == 'NN':
            detector = Anomaly_Detector(method, data.X, radius, data_window, time_window = buffer_window, percent = attacker_percentage, NN = num_neighbors, NNdefense_on = True)

        if defense_method == 'none':
            detector = Anomaly_Detector(method, data.X, radius, data_window, NNdefense_on = False)

        normal_pts = []
        iterations = 0
        while not detector.classify_point(target):
            iterations += 1
            print iterations
            old_center = detector.get_center()
            if random.random() < attacker_percentage:
                if method == 'random-out' or method == 'average-out':
                    new_point = simple_attack(detector.get_data(), target, detector.get_r())
                else:
                    new_point = greedy_optimal_attack(detector.get_data(), target, detector.get_r())
            else:
                pt = data.generate_new_points(1)
                (true_value, new_point) = (pt[0], pt[1][0])
                normal_pts.append((new_point, true_value))

            new_point = np.reshape(new_point, (1, len(new_point)))
            detector.add_point(new_point)

            if defense_method == 'grid':
                new_center = detector.get_center()
                m.addLine(old_center, new_center)
                c, w = m.checkPoints(baseline, thresh)
                for cen in c:
                    detector.add_grid_center(cen)

            if detector.classify_point(target):
                print 'Target achieved'

        
        normal_pts = np.array(normal_pts)
        true_positive = 0

        for n in normal_pts:
            if detector.classify_point(n[0]) != n[1]:
                true_positive += 1

        avg_true_positive.append(float(true_positive)/float(len(normal_pts)))
        avg_iterations.append(iterations)

    avg_true_positive = np.array(avg_true_positive)
    avg_iterations = np.array(avg_iterations)

    print 'Avg iterations: ', np.mean(avg_iterations)
    print 'Avg true positive: ', np.mean(avg_true_positive)

if __name__ == '__main__':
    main()
