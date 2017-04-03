# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 02:13:59 2017

@author: frcheng
"""
import numpy as np

class mesh:
      def __init__(self,minX,maxX,minY,maxY,xNum,yNum,mode='train',window=200):
            self.lati = np.linspace(minX,maxX,xNum)
            self.longi = np.linspace(minY,maxY,yNum)            
            self.lines = []
            self.values = []
            self.mode = mode
            self.window=window            
            for i in range(len(self.lati)):
                  self.values.append([])
                  for j in range(len(self.longi)):
                        self.values[i].append(0)
                        
      def findNearest(self,x,y):
            xi = np.searchsorted(self.lati,x)
            yi = np.searchsorted(self.longi,y)
            return xi,yi
      def line_intersection(self,line1, line2):
          xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
          ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
      
          def det(a, b):
              return a[0] * b[1] - a[1] * b[0]
      
          div = det(xdiff, ydiff)
          if div == 0:
             raise Exception('lines do not intersect')
      
          d = (det(*line1), det(*line2))
          x = det(d, xdiff) / div
          y = det(d, ydiff) / div
          return x, y

      def addLine(self,p1,p2):
            line = [np.array(p1,dtype=np.float),np.array(p2,dtype=np.float)]
            dist = np.linalg.norm(line[0]-line[1])
            if self.mode == 'train':
                  for i in self.lines:
                        x,y = self.line_intersection(line,i)
                        if x<np.max(self.lati) and x>np.min(self.lati) and y<np.max(self.longi) and y>np.min(self.longi):
                              xi,yi = self.findNearest(x,y)
                              self.values[xi][yi] += dist                  
                  self.lines.append(line)
            else:
                  self.lines = self.lines[-self.window:]
                  for i in self.lines:
                        x,y = self.line_intersection(line,i)
                        if x<np.max(self.lati) and x>np.min(self.lati) and y<np.max(self.longi) and y>np.min(self.longi):
                              xi,yi = self.findNearest(x,y)
                              self.values[xi][yi] += dist                  
                  self.lines.append(line)
                  if len(self.lines) > self.window:
                        del self.lines[0]
      
      def percentage(self):
            base = self.values/np.sum(self.values)
            return base
                        
minX = 1
maxX = 10
minY = 1
maxY = 10
xFine = 10
yFine = 10

m = mesh(minX,maxX,minY,maxY,xFine)

for i in training:
      m.addLine(i)

baseline = m.percentage

m.mode = 'test'
for i in test:
      m.addLine(i)
      new = m.percentage
      diff = new - baseline
      ## do something to centroid detector