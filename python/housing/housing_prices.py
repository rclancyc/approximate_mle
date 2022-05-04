import os, sys

sys.path.append('/Users/clancy/repos/approximate_mle/python')
sys.path.append('/Users/clancy/repos/approximate_mle/')


import numpy as np
import pandas as pd
from  loglikelihood import loglikelihood
import matplotlib.pyplot as plt
from scipy import optimize, stats
np.seterr(all="ignore")


# total least squares function definition
def TLS(A,b):
    m, _ = A.shape
    C = np.hstack((A,b.reshape(m,1)))
    
    # unlike matlab, numpy returns V.T
    _, _, Vt = np.linalg.svd(C)

    # take last row (smallest singular vector)
    v = Vt[-1,:]
    x_tls = -(1/v[-1])*v[:-1]

    return x_tls


df = pd.read_csv('house-prices-advanced-regression-techniques/full.csv')

df = pd.read_csv('house-prices-advanced-regression-techniques/full.csv')
df['totLivArea'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['GrLivArea']

# Focus on residential properties in low density areas and eliminate extreme quantile
myidx = df['MSZoning'] == 'RL'
df = df.loc[myidx,:]
myidx = df.LotArea < df.LotArea.quantile(0.99) 
df = df.loc[myidx,:]
myidx = df.totLivArea < df.totLivArea.quantile(0.99) 
df = df.loc[myidx,:]
myidx = df.SalePrice < df.SalePrice.quantile(0.99)
df = df.loc[myidx,:]


# Focus on the following features that are numeric values
tb = df.loc[:, ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'totLivArea', 'SalePrice']]

# Drop rows that have N/A values
tb.dropna(axis=0,how='any', inplace=True)

# Reset indices so they are more sensible and write as a numpy array
tb.reset_index(inplace=True, drop=True)
TB = np.asarray(tb)


# Extract features and prices (use centered prices not original ones)
features = TB[:,0:6]
orig_prices = TB[:,-1]
prices = orig_prices - np.mean(orig_prices)

# Round data where desired to conform to typical setup (other variables are assumed to be uncertain)
A = TB[:,0:6]
A[:,0] = np.round(A[:,0], -1)
A[:,1] = np.round(A[:,1], -2)
A[:,4] = A[:,4] + 0.5
A[:,5] = np.round(A[:,5], -2)

# Center the feature data 
A = A - np.mean(A, axis=0)

# Add column of ones...ways to correct for this, but I don't think any method will work for all instances
A = np.hstack((np.ones((prices.shape[0],1)),A))

# Set up parameter vectors
dists = ['constant', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'normal']
params = [None, [5], [50], [0.5], [0.5], [1], [50], [950000000]]

# Use the following arrangements if we want to drop the "OverallCond" feature
#dists = ['constant', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'normal']
#params = [None, [5], [50], [0.5], [1], [50], [900000000]]


np.random.seed(0)

numSims = 50
m = A.shape[0]
ols_err = np.zeros(numSims)
tls_err = np.zeros(numSims)
mle_err = np.zeros(numSims)
ols_err_med = np.zeros(numSims)
tls_err_med = np.zeros(numSims)
mle_err_med = np.zeros(numSims)
ols_err1 = np.zeros(numSims)
tls_err1 = np.zeros(numSims)
mle_err1 = np.zeros(numSims)

mSub = 10

for i in range(numSims):
    print('Simulation ', i)
    perms = np.random.permutation(prices.shape[0])
    idx = perms[0:mSub]
    nidx = perms[mSub:]
    B = A[idx,:]
    y = prices[idx]
    
    # Solve ols and tls
    xols = np.linalg.lstsq(B,y, rcond=None)[0]
    xtls = TLS(B, y)
    
    # Set initial iterate to zero, initialize MLE, 
    x0 = np.zeros(xols.shape[0])
    lf = loglikelihood(B, y, dists, params)
    neg_ll = lambda z: lf.negative_loglikelihood(z)
    results = optimize.minimize(neg_ll, xols*0.5, method='L-BFGS-B', jac=True)
    xmle = results.x
    
    # Track error metrics 
    # We look at 1 norm error
    C = A[nidx,:]
    z = prices[nidx]
    
    ols_err1[i] = np.linalg.norm(C@xols-z,1)/m
    tls_err1[i] = np.linalg.norm(C@xtls-z,1)/m
    mle_err1[i] = np.linalg.norm(C@xmle-z,1)/m
    
    ols_err[i] = np.linalg.norm(C@xols-z)/m
    tls_err[i] = np.linalg.norm(C@xtls-z)/m
    mle_err[i] = np.linalg.norm(C@xmle-z)/m
    
    ols_err_med[i] = np.median(np.abs(C@xols-z))
    tls_err_med[i] = np.median(np.abs(C@xtls-z))
    mle_err_med[i] = np.median(np.abs(C@xmle-z))
    
    print('Mean OLS1 error', round(ols_err1[i]))
    print('Mean MLE1 error', round(mle_err1[i]))
    print('Median OLS error', round(ols_err_med[i]))
    print('Median MLE error', round(mle_err_med[i]))




mle_err1temp = mle_err1[mle_err1 < np.quantile(mle_err1, .99)]
ols_err1temp = ols_err1[ols_err1 < np.quantile(ols_err1, .99)]
tls_err1temp = tls_err1[tls_err1 < np.quantile(tls_err1, .90)]




plt.rcParams['text.usetex'] = True
plt.figure()

plt.figure(figsize=(5,5))
bins = np.linspace(15000,110000,1000);
#plt.ylim([0,250])
plt.grid()
ax = plt.gca()
kernel1 = stats.gaussian_kde(mle_err1temp, bw_method=.6)#'silverman');
kernel2 = stats.gaussian_kde(ols_err1temp, bw_method=.4)#'silverman');#,ols_err1,tls_err1])
kernel3 = stats.gaussian_kde(tls_err1temp, bw_method=0.12)#'silverman')#'silverman');#,ols_err1,tls_err1])
plt.plot(bins, kernel1(bins), label=r'\bf{AML (proposed)}', ls='-', color='black', lw=3);
plt.plot(bins, kernel2(bins), label=r'\bf{OLS}',ls='--', color='goldenrod', lw=3);
plt.plot(bins, kernel3(bins), label=r'\bf{TLS}',ls=':', color='maroon', lw=3);
plt.xlabel(r'\bf{Mean absolute deviation error}', fontsize=16)
plt.ylabel(r'\bf{Approximate Density}',fontsize=16)

plt.xlim([15000, 110000])
#plt.legend()


parameters = {'axes.labelsize': 100,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
             'axes.titlesize': 20}
plt.rcParams.update(parameters)
plt.savefig('approx_densities.pdf') 

#plt.title(r'\bf{Mean abs. error by model}')

plt.xlim([16000, 94000])
plt.ylim([0,4.5e-5])
plt.legend()

plt.savefig('approx_density_for_paper.pdf') 




plt.rcParams['text.usetex'] = True

minexp = -1
maxexp = .25
bins = np.logspace(minexp, maxexp,29)
#bins = np.linspace(0.1,1.5,51)
plt.figure(figsize=(12,5))

#print(bins)
plt.subplot(1,2,1)
plt.xscale('log')
bins = np.logspace(-1, maxexp,27)
plt.hist((mle_err1/ols_err1), bins=bins)
plt.grid(which='both')
plt.axvline(x=1, color='red', linewidth=3)
plt.xlim([10**minexp,10**maxexp])
plt.ylim([0,200])
ax = plt.gca()
ax.yaxis.set_ticklabels([])
plt.xlabel(r"\bf{AML over OLS}", fontsize=20)
plt.ylabel(r'\bf{Frequency}',fontsize=20)
#print(bins)
plt.subplot(1,2,2)
plt.xscale('log')
minexp = -2
maxexp = .25
bins = np.logspace(minexp, maxexp,27)
plt.hist((mle_err1/tls_err1), bins=bins)
plt.grid(which='both')
plt.xlim([10**minexp,10**maxexp])
plt.axvline(x=1, color='red', linewidth=3)
plt.xlabel(r"\bf{AML over TLS}", fontsize=20)
plt.ylim([0,200])
#plt.ylabel(r'\bf{Frequency}',fontsize=20)
ax = plt.gca()
ax.yaxis.set_ticklabels([])


parameters = {'axes.labelsize': 100,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
          'axes.titlesize': 20}
plt.rcParams.update(parameters)
#plt.suptitle(r'\bf{Error ratio}', fontsize=24)

plt.savefig('err_ratio_for_paper.pdf') 
    

plt.show()