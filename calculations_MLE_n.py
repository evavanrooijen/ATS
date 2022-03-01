import numpy as np
from scipy import optimize

## returns 5000 values of THT using MLE paramters n0_h, n1_h
## seed = 1
## bias -0.27436752689701205
## variance 1.641074216963575
## MSE 1.7163517567791577

## seed = 2
##bias -0.29409420553716625
## variance 1.6946836688153752
## MSE 1.7811750705459122

## seed = 123
## bias -0.20142526800559235
## variance 1.3248797758274202
## MSE 1.3654519144185449

n = 100
n0 = 0.01
n1= 1
a0 = 4
a1 = 10
b0 = 2
b1 = 3

R = 500
est_n_h = np.ones(R)
# generate data
np.random.seed(2)

for r in range(R):
    # L follows normal (0,1)
    L = np.random.normal(loc=0, scale=1, size=n)
    # A follows Logistic
    epsGL = np.random.logistic(loc=0, scale=1, size=n)
    v0 = np.random.normal(loc=0, scale=1, size=n)
    v1 = np.random.normal(loc=0, scale=1, size=n)
    f = np.ones(n)
    A = np.ones(n)
    for i in range(n):
        # calculate A
        eval = n0+n1*L[i]+epsGL[i]
        if eval>0:
            A[i] = 1
        else:
            A[i] = 0

    # estimate propensity score using MLE
    def function(params): # TODO: should also take n, L, epsGL, define outside loop
        # params is numpy array of 2x1
        n0_h, n1_h = params
        logL = 0
        for i in range(n):
            logL = logL + A[i]* np.log(1-1/(1+np.exp(n0_h+n1_h*L[i]))) + (1-A[i])*np.log(1/(1+np.exp(n0_h+n1_h*L[i])))
        return -1*logL # max -> min

    result = optimize.minimize(function, [0, 1],  method = 'Nelder-Mead')
    if result.success:
        fitted_params = result.x
    else:
        #raise ValueError(result.message)
        print(result.message)
        continue
    n0_h, n1_h = fitted_params

    for i in range(n):
        # calculate propensity score using MLE parameters
        f[i] = 1-1/(1+np.exp(n0_h+n1_h*L[i]))

    # generate Y
    Y0 = a0 + b0*L + v0
    Y1 = a1 + b1*L + v1
    Y = (1-A)*Y0 + A*Y1

    # compute estimators
    sum = 0
    for i in range(n):
        if A[i] == 0:
            sum = sum - Y[i]/(1-f[i])
        elif A[i] ==1:
            sum = sum + Y[i]/f[i]

    Tn_h = sum/n
    est_n_h[r] = Tn_h

# calc bias
bias_n = np.mean(est_n_h) - (a1 -a0)
print(f'bias {bias_n}')

# calc var
var_n = np.mean((est_n_h - np.mean(est_n_h))**2)
print(f'variance {var_n}')

# calc MSE
MSE_n = bias_n**2 + var_n
print(f'MSE {MSE_n}')
