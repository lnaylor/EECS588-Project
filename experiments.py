#!/usr/bin/env python

import numpy as np
import random
import matplotlib as plt
from generate_data import Dataset
from defense import mesh
from circle_method import *

def get_new_num():
    get_new_num.counter += 1
    return str(get_new_num.counter)
get_new_num.counter = 0

def main():

    file_name = 'simpleattack-random-NNdefense'
    num_trials = 1
    method = 'random-out'
#    method = 'nearest-out'

#    defense_method = 'grid'
    defense_method = 'NN'
#    defense_method = 'none'
    
    radius = 2.45
    target = [3, 1]
    attacker_percentage = .25
    max_iterations = 10000
    plot_num = max_iterations/10
    
    data_window = 200
    buffer_window = 50
    num_neighbors = 10

    avg_precision = []
    avg_recall = []
    avg_false_positives = []
    avg_false_negatives = []
    avg_iterations = []

    for tr in range(num_trials):

        data = Dataset(p=2, n=10000, phi=0.05)
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

        m = mesh(min_x, max_x, min_y, max_y, xFine, yFine, radius=.05)
        

        if defense_method == 'grid':
            detector = Anomaly_Detector(method, data.X[:1], data.Y[:1], radius, data_window, NNdefense_on = False)
            for t in data[1:]:
                old_center = detector.get_center()
                detector.add_point(np.reshape(t.X, (1, len(t))), t.Y)
                new_center = detector.get_center()
                m.addLine(old_center, new_center)
                
            detector.set_grid_width(m.width)
            baseline = m.percentage()
            m.center = detector.get_center()
            m.mode = 'test'
        
        if defense_method == 'NN':
            detector = Anomaly_Detector(method, data.X, data.Y, radius, data_window, time_window = buffer_window, percent = attacker_percentage, NN = num_neighbors, NNdefense_on = True)

        if defense_method == 'none':
            detector = Anomaly_Detector(method, data.X, data.Y, radius, data_window, NNdefense_on = False)

        normal_pts = []
        iterations = 0
        for k in range(max_iterations):
            iterations += 1
            print iterations
            old_center = detector.get_center()
            if random.random() < attacker_percentage:
                if method == 'random-out' or method == 'average-out':
                    new_point = simple_attack(detector.get_data(), target, detector.get_r())
                else:
                    new_point = greedy_optimal_attack(detector.get_data(), target, detector.get_r())
                new_point = np.reshape(new_point, (1, len(new_point)))
                detector.add_point(new_point, 2)
            else:
                pt = data.generate_new_points(1)
                (true_value, new_point) = (pt[0], pt[1][0])
                normal_pts.append((new_point, true_value))
                new_point = np.reshape(new_point, (1, len(new_point)))
                detector.add_point(new_point, true_value)


            if defense_method == 'grid':
                new_center = detector.get_center()
                m.addLine(old_center, new_center)
                c, w = m.checkPoints(baseline, thresh)
                for cen in c:
                    detector.add_grid_center(cen)

                rec = detector.get_grid_centers()
                w = detector.get_grid_width()
                w = w/2
                line = [old_center, new_center]
                recs = []
                for r in rec:
                    x1 = r[0]-w
                    x2 = r[0]+w
                    y1 = r[1]-w
                    y2 = r[1]+w
                    recs.append([[x1,y1],[x2,y2]])

                should_delete = False
                for r in recs:
                    if m.line_rec_intersection(line, r):
                        should_delete = True

                if should_delete:
                    detector.remove_point()
            
            if (k%plot_num) == 0:

                fig = plt.figure()

                ax = fig.add_subplot(1, 1, 1)
                circ = plt.Circle(detector.get_center(), radius, fill = False)
                ax.add_patch(circ)
                
                plt.scatter(target[0], target[1], color = 'g', marker = 'x', s=35, label = 'target point')
                plt.scatter(detector.get_data()[ detector.get_labels() == True, 0], detector.get_data()[ detector.get_labels()== True, 1], color = 'r', marker = 'o', s=10, label = 'anomalous points')
                plt.scatter(detector.get_data()[detector.get_labels()== False, 0], detector.get_data()[detector.get_labels()== False, 1], color = 'k', marker = '+', s=20, label = 'nominal points')
                if attacker_percentage > 0:
                    plt.scatter(detector.get_data()[detector.get_labels()== 2, 0], detector.get_data()[detector.get_labels()== 2, 1], color = 'b', marker = '^', s=10, label = 'attack points')

                plt.xlim([-5, 5])
                plt.ylim([-5, 5])
                plt.gca().set_aspect('equal', adjustable='box')
#                plt.axis('equal')
#                plt.legend(loc=0)
                fig.savefig(file_name+get_new_num()+'.png', bbox_inches='tight')
                
#                plt.show()

            if detector.classify_point(target):
                print 'Target achieved'
                print detector.get_center()
                break

        
        normal_pts = np.array(normal_pts)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        X = []
        Y = []
        true_y = []
        for n in normal_pts:
            X.append(n[0])
            Y.append(detector.classify_point(n[0]))
            true_y.append(n[1])
            if detector.classify_point(n[0]) and not n[1]:
                true_negative += 1
            if detector.classify_point(n[0]) and n[1]:
                false_negative += 1
            if not detector.classify_point(n[0]) and not n[1]:
                false_positive += 1
            if not detector.classify_point(n[0]) and n[1]:
                true_positive += 1
        
        print 'Number of iterations: ', iterations
        print 'True positive: ', true_positive
        print 'True negative: ', true_negative
        print 'False positive: ', false_positive
        print 'False negative: ', false_negative

        avg_iterations.append(iterations)
        avg_recall.append(float(true_positive)/(true_positive+false_negative))
        avg_precision.append(float(true_positive)/(true_positive+false_positive))
        avg_false_positives.append(float(false_positive)/(false_positive+true_negative))
        avg_false_negatives.append(float(false_negative)/(false_negative+true_positive))

    avg_iterations = np.array(avg_iterations)
    avg_recall = np.array(avg_recall)
    avg_precision = np.array(avg_precision)
    avg_false_positives = np.array(avg_false_positives)
    avg_false_negatives = np.array(avg_false_negatives)

    print 'Average iterations: ', np.mean(avg_iterations)
    print 'Average recall: ', np.mean(avg_recall)
    print 'Average precision: ', np.mean(avg_precision)
    print 'Average false positives: ', np.mean(avg_false_positives)
    print 'Average false negatives: ', np.mean(avg_false_negatives)
    
#    fig = plt.figure()
#
#    ax = fig.add_subplot(1, 1, 1)
#    circ = plt.Circle(detector.get_center(), radius, fill = False)
#    ax.add_patch(circ)
#    
#    plt.scatter(target[0], target[1], color = 'g', marker = 'x', s=35, label = 'target point')
#    plt.scatter(detector.get_data()[ detector.get_labels() == True, 0], detector.get_data()[ detector.get_labels()== True, 1], color = 'r', marker = 'o', s=10, label = 'anomalous points')
#    plt.scatter(detector.get_data()[detector.get_labels()== False, 0], detector.get_data()[detector.get_labels()== False, 1], color = 'k', marker = '+', s=20, label = 'nominal points')
#    if attacker_percentage > 0:
#        plt.scatter(detector.get_data()[detector.get_labels()== 2, 0], detector.get_data()[detector.get_labels()== 2, 1], color = 'b', marker = '^', s=10, label = 'attack points')
#
#    plt.axis('equal')
#    fig.savefig(file_name+get_new_num()+'.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
