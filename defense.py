# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 02:13:59 2017

@author: frcheng
"""
import numpy as np

class mesh:
      def __init__(self,minX,maxX,minY,maxY,xNum,yNum,mode='train',window=200,center = None, radius=0,searchSpace=False):
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
            self.width = float(maxX - minX)/float(xNum-1)
            self.center = center
            self.r = radius
            
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
                  x = None
                  y = None
      
            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y
      def line_rec_intersection(self,line,rec):
            xrec = np.sort(rec,axis=0)
            xInit = xrec[0][0] - 1.0
            xEnd = xrec[-1][0] + 1.0
            yrec = np.sort(rec,axis=1)
            yBot = yrec[0][1]
            yTop = yrec[1][1]
            m = (line[1][1] - line[0][1])/(line[1][0]-line[0][0])
            y0 = line[1][1] - m*line[1][0]
            
            def yLine(x):
                  return m*x + y0
            yInit = yLine(xInit)
            yEnd = yLine(xEnd)
            
            x1 = rec[0][0]
            x2 = rec[1][0]
            y1 = rec[0][1]
            y2 = rec[1][1]
            
            def F(x,y):
                  return (yEnd - yInit)*x + (xInit - xEnd)*y + (xEnd*yInit - xInit*yEnd)
            
            check  = []
            check.append( F(x1,y1) )
            check.append( F(x2,y1) )
            check.append( F(x1,y2) )
            check.append( F(x2,y2) )
            
            if all(i > 0 for i in check) or all(i < 0 for i in check):
                  return False
            elif yEnd > yTop and yInit > yTop:
                  return False
            elif yInit < yBot and yEnd < yBot:
                  return False
            else:
                  return True
                        
      def addLine(self,p1,p2):
            line = [np.array(p1,dtype=np.float),np.array(p2,dtype=np.float)]
#            dist = np.linalg.norm(line[0]-line[1])
            if self.mode == 'train':
                  for i in self.lines:
                        x,y = self.line_intersection(line,i)
                        if x is None:
                              continue
                        elif x<np.max(self.lati) and x>np.min(self.lati) and y<np.max(self.longi) and y>np.min(self.longi):
                              xi,yi = self.findNearest(x,y)
                              self.values[xi][yi] += 1.                  
                  self.lines.append(line)
            else:
                  self.lines = self.lines[-self.window:]
                  for i in self.lines:
                        x,y = self.line_intersection(line,i)
                        if x is None:
                              continue
                        elif x<np.max(self.lati) and x>np.min(self.lati) and y<np.max(self.longi) and y>np.min(self.longi):
                              xi,yi = self.findNearest(x,y)
                              self.values[xi][yi] += 1.                  
                  self.lines.append(line)
                  if len(self.lines) > self.window:
                        del self.lines[0]
      
      def percentage(self):
            base = self.values[:]
            if self.center is None:
                  maxAll = np.argmax(base)
                  x, y = maxAll/len(self.lati),maxAll % len(self.longi)
                  base[x][y] = 0
                  self.center = (self.lati[x],self.longi[y])
            for i in range(len(self.lati)):
                  for j in range(len(self.longi)):
                        pt = np.asarray([self.lati[i],self.longi[j]])
                        if np.linalg.norm(pt-self.center) < self.r:
                              base[i][j] = 0
            base = base/np.sum(base)
            return base
        
      def checkPoints(self,baseline,thresh):
            ctrs = []
            diff = self.percentage() - baseline
            for i in range(len(self.lati)):
                  for j in range(len(self.longi)):
                        if diff[i][j] > thresh:
                              ctrs.append((self.lati[i],self.longi[j]))
            return ctrs,self.width


def main():
    #n = mesh(1,10,1,10,10,10)
    #baseline = n.percentage()
    #n.addLine((1,2),(2,3))
    #n.addLine((1,3),(2,2))
    #c,w = n.checkPoints(baseline,0.5)
    
                            
    minX = 1
    maxX = 10
    minY = 1
    maxY = 10
    xFine = 10
    yFine = 10
    thresh = 0.5
    
    m = mesh(minX,maxX,minY,maxY,xFine,yFine)
    
    
    for i in training:
          #before = 
          #after = 
          m.addLine(i)
    
    baseline = m.percentage()
    
    m.mode = 'test'
    for i in test:
          #before = 
          #after =       
          m.addLine(i)
          c,w = m.checkPoints(baseline,thresh)
          ## do something to centroid detector
          # 1) threshold and classify all in grid as anomalous
          # 2) smooth and check gradient in addition to threshold
if __name__=='__main__':
    main()
