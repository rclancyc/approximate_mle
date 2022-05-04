#from Users/clancy/repos/approximate_mle/python/regression_test import give_cgf
import numpy as np
import copy

class loglikelihood:
    def __init__(self, H, y, distributions, parameters):
        """
        :params H:  (m x n) numpy array of design matrix
        :params y:  (n x []) or (n x 1) numpy array of respons variable
        :params distributions:  n+1 element list (or n+1 element list of m element lists) of distribution types for the 
                                different features (features that differ by row). Extra element is for additive distribution
        :params parameters: n+1 element list of distribution parameters lists (or n+1 element list of m element 
                            lists of distribution parameter lists). Mos
        """
        # load problem details such as nominal matrix, response variables, densities for the features
        self.H = H                          # store given design matrix 
        self.y = y                          # store right hand side 
        self.distributions = distributions  # store list or matrix of named distributions, i.e. normal, uniform etc
        self.parameters = parameters        # store corresponding distribution parameters
        self.m = H.shape[0]                 # store number of rows in design matrix
        self.n = H.shape[1]                 # store number of columns in design matrix
        self.last_x = None                  
        self.last_t = None
        self.last_ells = None

    
    def func_and_grad(self, xs):
        self.last_x = xs                    # store point at which we evaluate
        ell = 0                             # initialize approx. log likelihood
        nterms = len(self.distributions)    # how many regression parameters are we looking for
        ts = np.zeros((self.m,))            # initial ts which are solution to saddle point equation, one for each row
        
        # initialize vectors to store CGF and its derivatives
        Kps = np.zeros((self.m,))           
        Kpps = np.zeros((self.m,))
        Kppps = np.zeros((self.m,))
        dK_dxs = np.zeros((self.m, self.n))
        dKp_dxs = np.zeros((self.m, self.n))
        dKpp_dxs = np.zeros((self.m, self.n))

        # loop through rows of H to construct log likelihood function
        for i in range(self.H.shape[0]):
            # initialize poles list and lists that store function for a particular row of H
            poles = []  
            cgf_temp_list = []

            # loop through all columns of H AND the additive noise portion as well
            for j in range(nterms):
                # get parameters and distribution for the current feature (or additive noise terms)
                if isinstance(self.distributions[j], list):
                    dist = self.distributions[j][i]
                    params = self.parameters[j][i]    
                else:
                    dist = self.distributions[j]
                    params = self.parameters[j]
                
                # for additive noise, set a = 0 and x = 1 since this is unscaled and A[i,j] = 0 is unshifted 
                if j < self.n:
                    h = self.H[i,j]
                    x = xs[j]
                else: 
                    h = 0
                    x = 1

                # construct cumulant generating functions and its derivatives for current columns (in the row)
                curr_cgf = lambda t, di_=dist, par_=params, h_=h, x_=x: self.cgf(di_, par_, h_, x_, t)
                cgf_temp_list.append(curr_cgf)

                # for newtons method, must track poles, do so below
                if dist == 'laplace':
                    poles.extend([params[0]/x, -params[0]/x])
                if dist == 'exponentially_clipped':
                    poles.append(np.sign(h)*params[0]/x)
                

            # pass list of CGFs for current row and t to get a tuple cgf with K, K', K'', K'''
            cumulant_fun = lambda t: get_cgf_sum(cgf_temp_list, t)

            # extract CGF (or its derivative) for use in Newton's method/function evaluation
            K = lambda t: cumulant_fun(t)[0]
            Kp = lambda t: cumulant_fun(t)[1]
            Kpp = lambda t: cumulant_fun(t)[2]
            Kppp = lambda t: cumulant_fun(t)[3]
            Kp_minus_y = lambda t: Kp(t) - self.y[i]
            
            # turn poles into a numpy array and find the poles that bracket zero
            poles = np.asarray(poles)
            posp = np.min(poles[poles>0]) if poles[poles>0].shape[0] > 0 else np.inf
            negp = np.max(poles[poles<0]) if poles[poles<0].shape[0] > 0 else -np.inf
            a0 = -9e6
            b0 = 10e6

            # solve saddle point equation using data for the ith point and store in a vector
            t_y = newton_safe(Kp_minus_y, Kpp, a0, b0, [negp, posp])
            ts[i] = t_y

            # now that we found t_y, we can loop through an compute the matrices
            for j in range(self.n):                
                dK_dxs[i, j] = cgf_temp_list[j](t_y)[4]
                dKp_dxs[i, j] = cgf_temp_list[j](t_y)[5]
                dKpp_dxs[i, j] = cgf_temp_list[j](t_y)[6]
            # get contribution to log likelihood function from the ith row 
            ell_i = K(t_y)  - 0.5*np.log(Kpp(t_y)) - t_y*self.y[i]

            # store to generate derivative components
            Kps[i] = Kp(t_y)
            Kpps[i] = Kpp(t_y)
            Kppps[i] = Kppp(t_y)

            # accumulate log likelihood for current row to full likelihood function
            ell = ell + ell_i

        # calculate derivatives used to find gradient
        dq_dt_inv = np.diag(1/Kpps)
        dl_dx = np.dot(np.ones((1,self.m)),(dK_dxs - 0.5*np.dot(dq_dt_inv,dKpp_dxs)))
        dl_dt = Kps - self.y - 0.5*(Kppps/Kpps)
        dq_dx = dKp_dxs

        grad_ell = dl_dx - np.dot(np.dot(dl_dt,dq_dt_inv),dq_dx)

        return ell, grad_ell.reshape(xs.shape)

    def negative_loglikelihood(self, xs):
        F = self.func_and_grad(xs)
        return -F[0], -F[1]

        
    def cgf(self, distribution, parameters, h, x, t):
        
        ### DEGENERATE/INTERCEPT PARAMETER
        if distribution == 'constant':
            K = h*x*t
            Kp = h*x
            Kpp = 0
            Kppp = 0
            dK_dx = h*t
            dKp_dx = h
            dKpp_dx = 0

        ### UNIFORM 
        if distribution == 'uniform':
            if len(parameters) != 1:
                print('Should only have half half interval length as a parameter for uniform')
            else:
                delta = parameters[0]

            k = delta*x
            if abs(k*t) < 0.01:
                # use taylor expansion when k*t is small enough 
                K = h*t*x + (k*t)**2/6 - (k*t)**4/72 + (k*t)**6/648
                Kp = h*x + (k**2)*t/3 - (k**4)*(t**3)/18 + (k**6)*(t**5)/108
                Kpp = (k**2)/3 - (k**4)*(t**2)/6 + 5*(k**6)*(t**4)/108
                Kppp = -(k**4)*t/3 + 5*(t**6)*(t**3)/27

                b = delta*t 
                dK_dx = h*t + (b**2)*x/3 - (b**4)*(x**3)/18 + (b**6)*(x**5)/108
                dKp_dx = h + 2*delta*(x*t)/3 - 2*(delta**4)*(x*t)**3/9 + 6*(delta**6)*(x*t)**5/108
                dKpp_dx = 2*(delta**2)*x/3 - 4*(delta**4)*(x**3)*(t**2)/6 + 30*(delta**6)*(x**5)*(t**4)/108
            else:
                K = h*t*x + np.log(np.sinh(k*t)/(k*t))
                Kp = h*x - 1/t + k/np.tanh(k*t)
                Kpp = 1/t**2 - (k/np.sinh(k*t))**2
                Kppp = -2/t**3 + 2*(k**3)/(np.tanh(k*t)*np.sinh(k*t)**2)

                dK_dx = h*t + (delta*t)/np.tanh(k*t) - 1/x
                dKp_dx = h + delta/np.tanh(k*t) - (t*x*(delta**2))/(np.sinh(k*t)**2)
                dKpp_dx = 2*t*(x**2)*(delta**3)/((np.sinh(k*t)**2)*np.tanh(k*t)) - 2*x*(delta**2)/(np.sinh(k*t)**2)

        ### NORMAL
        if distribution == 'normal':
            if len(parameters) != 1:
                print('Should only have variance as a parameter for normal')
            else:
                sigma_sq = parameters[0]
        
            K = h*t*x + 0.5*sigma_sq*(t**2)*(x**2)
            Kp = h*x + sigma_sq*(x**2)*t
            Kpp = sigma_sq*(x**2)
            Kppp = 0

            dK_dx = h*t + sigma_sq*(t**2)*x
            dKp_dx = h + 2*sigma_sq*t*x
            dKpp_dx = 2*sigma_sq*x

        ### LAPLACE  ( we have an error in Laplace case)
        if distribution == 'laplace':
            if len(parameters) != 1:
                print('Should only have a rate as a parameter for laplace distribution')
            else:
                # let beta be the rate
                beta = parameters[0]
            
            pl = beta + x*t
            mi = beta - x*t 

            #K = h*t*x + 2*np.log(beta) + np.log(mi) - np.log(pl)
            K = h*t*x + 2*np.log(beta) - np.log(beta**2 - (x*t)**2)
            Kp = h*x + x/mi - x/pl
            Kpp = x**2/mi**2 + x**2/pl**2
            Kppp = 2*x**3/mi**3 - 2*x**3/pl**3
            
            dK_dx = h*t + t/mi - t/pl
            dKp_dx = h + 1/mi - 1/pl + t*x/pl**2 + t*x/mi**2
            dKpp_dx = 2*x*(1/mi**2 + 1/pl**2) +2*t*x**2 * (1/mi**3 - 1/pl**3)
 

        ### EXPONENTIALLY CLIPPED
        if distribution == 'exponentially_clipped':
            if len(parameters) != 1:
                print('Should have two parameters for exponential clipped ')
            else:
                # let beta be the rate
                beta = parameters[0]
                threshold = parameters[1]
            
            # q indicates where or not the threshold is met
            a = 1 if abs(h) == threshold else 0
            s = np.sign(h)

            K =  h*x*t - a*np.log(1-s*t*x/beta)
            Kp = h*x + a*s*x/(beta - s*t*x)
            Kpp = a*(s*x)**2/(beta - s*t*x)**2
            Kppp = 2*a*(s*x)**3/(beta - s*t*x)**3

            mi = s*t*x - beta
            dK_dx = h*t + a*s*t/mi
            dKp_dx = h - a*s/mi + a*t*x*s**2/mi**2
            dKpp_dx = 2*a*(x/mi*2 - s*t*x**2/mi**3)

        return K, Kp, Kpp, Kppp, dK_dx, dKp_dx, dKpp_dx







def get_cgf_sum(functions, t):
    temp = np.zeros((len(functions), 7))
    for i, fun in enumerate(functions):
        temp[i,:] = fun(t)

    temp2 = np.sum(temp, axis=0)
    return tuple(temp2)




def newton_safe(f, df, a0, b0, poles):
    """myroot = newton_safe(f, df, a, b)
    
    INPUTS
    f: function handle on which to find root
    df: function gradient handle
    a: left bracket for bounding
    b: right bracket for bounding
    
    OUTPUTS
    myroot: a root for function f
    
    This function performs a bracketed Newton Method that falls back to
    the Bisection method if a step is taken outside of the current
    interval for which a root exists. This particular implementation acts
    on a vector valued function that has a diagonal Jacobian and is
    purpose built finding t(x) in the approximate MLE problem
    """

    a = poles[0]+1e-5 if poles[0] > a0 else a0
    b = poles[1]-1e-5 if poles[1] < b0 else b0
    tol = 1.0e-7  # must be small for uniform with rounding error level noise

    maxit = 100
    root_found = False
    num_exp = 0
    max_num_exp = 20
    t = 0.5*(a+b)
    t0 = t

    # set initial t based on bounds and small perturbation
    while not root_found:  
        it = 0
        ft = f(t)
        dft = df(t)
        while abs(ft) > tol and it < maxit:  
            it = it + 1
            #print('func val =', ft, 'at it =', it) if it % 20 == 0 else None
            a, b = bisect_interval(f,a,b)     # bracket interval each time to ensure progress
            t = t - ft/dft                     # take newton step       
            
            # did newton step take us outside safe brackets?
            if t < a or t > b:
                # use bisection step instead
                t = (a + b)/2
            ft = f(t)
            dft = df(t)

        myroot = t
        # can we find a root? If not  expand brackets (this should only happen for continuous CGFs)
        if abs(ft) > tol:
            tol = tol*5
            a = t0 - (t0-a0)*2**(num_exp+1)
            b = t0 + (b0-t0)*2**(num_exp+1)

            # make sure we don't expand brackets beyond poles
            a = poles[0] if poles[0] > a else a
            b = poles[1] if poles[1] < b else b

            print('Expanding brackets to', a, 'and', b, 'current ||f(t)|| =',ft)

            a0 = a
            b0 = b
            num_exp = num_exp + 1
            if num_exp > max_num_exp:
                print('Newton Safe violates tolerance of', tol, '||f(t)|| =', abs(ft))
                root_found = True
            else:
                t = 0.5*(a+b) + np.random.normal(0,1) 
        else:
            root_found = True
    
    return myroot



def bisect_interval(f, a, b):
    """
    a, b = bisect_interval(f, a, b)
    INPUTS
    f: function handle
    a: lower boundary for a hypercube containing a root of f
    b: upper boundary for a hypercube containing a root of f
    
    OUTPUTS
    a: new lower boundary after interval has been bisected
    b: new upper boundary after interval has been bisected
    """
    
    c  = 0.5*(a+b)
    if f(a)*f(c) >= 0:
        a = c
    else:
        b = c
        
    return a, b



def TLS(A,b):
    m, n = A.shape
    C = np.hstack((A,b.reshape(m,1)))
    
    # unlike matlab, numpy returns V.T
    _, _, Vt = np.linalg.svd(C)

    # take last row (smallest singular vector)
    v = Vt[-1,:]
    x_tls = -(1/v[-1])*v[:-1]

    return x_tls
