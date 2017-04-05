
from __future__ import print_function

import cplex
import numpy as np

def setProblemData(p, current, data, direction, r):
    #Set to maximization problem
    p.objective.set_sense(p.objective.sense.maximize)

    #Linear Constraints
    indices = p.variables.add(names = [str(k) for k in range(len(current))])
    for j in range(len(data)):
        my_rhs = np.dot(data[j],data[j]) - np.dot(current,current)
        lin_coeff =[]
        for k in range(len(current)):
            lin_coeff.append(2*(data[j][k]-current[k]))
        p.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = indices, val = lin_coeff)],
                                 rhs = [my_rhs], senses = ["L"])

    #Objective function
    for k in range(len(current)):
        p.objective.set_linear(k,direction[k])

    #Quadartic Constraint
    sumData = [sum(x) for x in zip(*data)]
    sumDataSquare = sum( j*j for j in sumData)
    q_rhs = r*r-(1./len(current))*sumDataSquare
    q_lin_coeff = [x * -(2./len(current)) for x in sumData]
    q_quad_coeff = [1.]*len(current)
    p.quadratic_constraints.add(lin_expr = cplex.SparsePair(ind = indices, val = q_lin_coeff),
                                quad_expr = cplex.SparseTriple(ind1=indices, ind2=indices, val=q_quad_coeff),
                                rhs=q_rhs, sense  = 'L')


def QCLP(data, direction, r):
    #Change data from array to list
    #list(data)
    data.tolist()
    
    #Set of potential new points and corresponding objective funciton values
    solution_new_pt = []
    solution_value = []
    
    #Find corresponding new point for every point in data
    for i in range(len(data)):
        p = cplex.Cplex()
        setProblemData(p,data[i],data,direction,r)
        p.solve()
        
        new_point =p.solution.get_values()
        value = np.dot(np.asarray(new_point)-np.asarray(data[i]),np.asarray(direction))
        solution_new_pt.append(new_point)
        solution_value.append(value)

        cplex.Cplex.end(p)

    #Find the new point and return
    new_index = solution_value.index(max(solution_value))
    new_point = np.asarray(solution_new_pt[new_index])
    return new_point


if __name__ == "__main__":
    data1 = [[1.0,2.0,3.0],[2.0,1.0,4.0]]
    data = np.array([[1.0,2.0,3.0],[2.0,1.0,4.0]])
    direction = [1.,1.,1.]
    r = 1.0
    QCLP(data, direction, r)
