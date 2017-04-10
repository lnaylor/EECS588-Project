
from __future__ import print_function

import cplex
import numpy as np

def setProblemData(p, current, data, direction, r):
    #Set to maximization problem
    p.objective.set_sense(p.objective.sense.maximize)
    
    
    #Linear Constraints
    indices = p.variables.add(names = [str(k) for k in range(len(current))])
    for j in range(len(data)):
        #if(np.asarray(current) != np.asarray(data[j]))
        #if not all(x in current for x in data[j]):
        #print('CURRENT' + str(current))
        #print('DATA{J}' + str(data[j]))
        my_rhs = np.dot(np.asarray(data[j]),np.asarray(data[j])) - np.dot(np.asarray(current),np.asarray(current))
        lin_coeff =[]
        for k in range(len(current)):
            lin_coeff.append(2.0*(data[j][k]-current[k]))
        p.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = indices, val = lin_coeff)],
                                 rhs = [my_rhs], senses = ["L"])
    
    #print('indices:  ' + str(indices))
    #print('lin_coeff:   ' + str(lin_coeff))
    #print('rhs:      '+ str(my_rhs))
    #print('number of linear contraints:  '+ str(p.linear_constraints.get_rows()))
    
    #Quadartic Constraint
    sumData = [sum(x) for x in zip(*data)]
    sumDataSquare = sum( j*j for j in sumData)
    q_rhs = r*r-(1./(len(data)*len(data)))*sumDataSquare
    q_lin_coeff = [x * -(2./len(data)) for x in sumData]
    q_quad_coeff = [1.]*len(current)
    p.quadratic_constraints.add(lin_expr = cplex.SparsePair(ind = indices, val = q_lin_coeff),
                                quad_expr = cplex.SparseTriple(ind1=indices, ind2=indices, val=q_quad_coeff),
                                rhs=q_rhs, sense  = 'L')

    #Objective function
    for k in range(len(current)):
        p.objective.set_linear(k,direction[k])
        offset = -np.dot(np.asarray(current),direction)
        p.objective.set_offset(offset)


def QCLP(data, direction, r):
    #Change data from array to list
    #list(data)
    data.tolist()
    
    #Set of potential new points and corresponding objective funciton values
    solution_new_pt = []
    solution_value = []
    data_indices = []
    
    #Find corresponding new point for every point in data
    for i in range(len(data)):
        p = cplex.Cplex()
        setProblemData(p,data[i],data,direction,r)
        p.set_log_stream(None)
        p.set_results_stream(None)
        p.solve()
        
        if p.solution.get_status() == 1:
            #print('CURRENT: '+str(data[i]))
            new_point =p.solution.get_values()
            #print('NEWWWWWW: '+str(new_point))
            value = p.solution.get_objective_value()
            #value = np.dot(np.asarray(new_point)-np.asarray(data[i]),np.asarray(direction))
            solution_new_pt.append(new_point)
            solution_value.append(value)
            data_indices.append(i)
        
        cplex.Cplex.end(p)
    
    #Find the new point and return
    new_index = solution_value.index(max(solution_value))
    new_point = np.asarray(solution_new_pt[new_index])
    #print('new point:   '+ str(new_point))
    #print('REPLACED POINT: '+ str(data[data_indices[new_index]]))
    #print('f REPLACED: '+ str(solution_value[new_index]))
    return new_point


if __name__ == "__main__":
    data1 = [[1.0,2.0,3.0],[2.0,1.0,4.0]]
    data = np.array([[1.0,2.0],[2.0,1.0],[3.,4.]])
    direction = [1.,1.]
    r = 1.0
    QCLP(data, direction, r)

