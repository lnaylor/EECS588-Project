#!/usr/bin/env python

import numpy as np
import random

class Anomaly_Detector:
    
    def __init__(self, method, data, r):
        self.__method = method
        self.__data = data
        self.__r = r
        self.__center = np.mean(data, axis=0)

    def add_point(self, point):
        if self.__method=="random-out":
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
            print "invalid removal method"
    

    def classify_point(self, point):
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

def main():

    method = 'random-out'
    data = np.random.normal(size=(50, 2))
    detector = Anomaly_Detector(method, data, 1.0)

    center = detector.get_center()
    print center
    print detector.classify_point([1, 1])

    while (not detector.classify_point([1,1])):
        
        detector.add_point([simple_attack(data, [1, 1], 1.0)])
        print detector.get_center()

    print detector.classify_point([1,1])


if __name__ == '__main__':
    main()
