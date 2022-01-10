import numpy as np
from loglikelihood import loglikelihood
from numpy import random as RA
from numpy import linalg as LA
from scipy import optimize


def test_normals():
    # THANK GOODNESS! NORMAL WORKS!!!
    dist_list = []
    param_list = []
    myvar1 = 2 
    H = RA.uniform(-10,10,(m,n)) # H is observed
    G = H + RA.normal(0,myvar1,(m,n))
    for j in range(n):
        dist_list.append('normal')    
        param_list.append([myvar1])
    y = G@x + RA.randn(m)
    myvar2 = 0.5
    dist_list.append('normal')
    param_list.append([myvar2])
    lf = loglikelihood(H, y, dist_list, param_list)
    f, g = lf.func_and_grad(x)
    f1, _ = lf.func_and_grad(x + eps*e1)
    f2, _ = lf.func_and_grad(x + eps*e2)
    approx_grad = (1/eps)*np.array([f1-f, f2-f])
    print(g[0])
    print(approx_grad)


def test_laplaces():
    # laplace seems to work
    dist_list = []
    param_list = []
    myvar1 = 1 
    H = RA.uniform(-10,10,(m,n)) # H is observed
    G = H + RA.laplace(0,myvar1,(m,n))
    for j in range(n):
        dist_list.append('laplace')    
        param_list.append([myvar1])
    y = G@x + RA.randn(m)
    myvar2 = 1
    dist_list.append('laplace')
    param_list.append([myvar2])
    lf = loglikelihood(H, y, dist_list, param_list)
    f, g = lf.func_and_grad(x)
    f1, _ = lf.func_and_grad(x + eps*e1)
    f2, _ = lf.func_and_grad(x + eps*e2)
    approx_grad = (1/eps)*np.array([f1-f, f2-f])
    print(g[0])
    print(approx_grad)


def test_uniform():
    dist_list = []
    param_list = []
    myvar1 = 1 
    H = RA.normal(10,4,(m,n))
    G = H + RA.uniform(0,myvar1,(m,n))
    for j in range(n):
        dist_list.append('uniform')    
        param_list.append([myvar1])
    y = G@x + RA.randn(m)
    myvar2 = 1
    dist_list.append('uniform')
    param_list.append([myvar2])
    lf = loglikelihood(H, y, dist_list, param_list)
    f, g = lf.func_and_grad(x)
    f1, _ = lf.func_and_grad(x + eps*e1)
    f2, _ = lf.func_and_grad(x + eps*e2)
    approx_grad = (1/eps)*np.array([f1-f, f2-f])
    print(g[0])
    print(approx_grad)




def test_uniform_and_laplace():
    # Getting errors here, focus on uniform first
    accuracy = 0                    # round to the "accuracy" place
    delta = 0.5*10**(-accuracy)     # this is the level of rounding requested
    # initialize regression problem 
    Htrue = np.random.uniform(-10, 10, (m,n))   # establish true H (using uniform here)
    H = np.round(Htrue, accuracy)               # round using accuracy to get observed design
    ytrue = np.dot(Htrue,x)                     # get the TRUE un-noised version of y
    y = ytrue + np.random.laplace(0,1,(m,))     # and Laplacian noise in this case
    dist_list = []
    param_list = []
    # set distributions for the regression problem 
    for j in range(n):
        dist_list.append('uniform')    
        param_list.append([delta])
    dist_list.append('laplace')
    param_list.append([1])
    lf = loglikelihood(H, y, dist_list, param_list)
    f, g = lf.func_and_grad(x)
    f1, _ = lf.func_and_grad(x + eps*e1)
    f2, _ = lf.func_and_grad(x + eps*e2)
    approx_grad = (1/eps)*np.array([f1-f, f2-f])
    print('Gradient', g[0])
    print('Apx grad', approx_grad)
    


def true_test():
    accuracy = 0                    # round to the "accuracy" place
    delta = 0.5*10**(-accuracy)     # this is the level of rounding requested
    
    # initialize regression problem 
    Htrue = np.random.uniform(-10, 10, (m,n))   # establish true H (using uniform here)
    H = np.round(Htrue, accuracy)               # round using accuracy to get observed design
    ytrue = np.dot(Htrue,x)                     # get the TRUE un-noised version of y
    y = ytrue + np.random.laplace(0,10,(m,))     # and Laplacian noise in this case
    dist_list = []                              # initialize distribution list
    param_list = []                             # initialize list of parameters
    
    # set distributions for the regression problem 
    for j in range(n):
        dist_list.append('uniform')    
        param_list.append([delta])
    dist_list.append('laplace')
    param_list.append([1/10])
    
    # instantiate log likelihood object
    lf = loglikelihood(H, y, dist_list, param_list)

    # define an anonymous function
    neg_ll = lambda z: lf.negative_loglikelihood(z)

    xls = LA.lstsq(H, y, rcond=1e-15)[0]    
    results = optimize.minimize(neg_ll, x0, method='L-BFGS-B', jac=True)

    print(results)
    print('LS   solution error', LA.norm(xls-x,       np.inf))
    print('AMLE solution error', LA.norm(results.x-x, np.inf))


m = 11                         # number of rows in matrix
n = 10                         # set number of columns (linear parameters to solve for)
#e1 = np.array([1,0])
#e2 = np.array([0,1])
#eps = 1.e-6

#np.random.seed(1)               # set random number seed
x = np.random.uniform(-10,10, (n,))           # linear paramater vector
x0 = RA.randn(n)

#test_normals()
#test_laplaces()
#test_uniform_and_laplace()
#test_uniform()
true_test()








