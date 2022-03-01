import numpy as np
from scipy import optimize

## returns 5000 values of THT using true paramters n0, n1
## seed = 1
## bias -0.012642522404788359
## variance 2.6073930959325
## MSE 2.6075529293052555

## seed = 2
## bias -0.0068979814601615175
## variance 2.5321826292582283
##MSE 2.532230211406453

## seed = 123
## bias 0.06416968158833658
## variance 2.5897697401151905
## MSE 2.593887488150339

n = 100
n0 = 0.01
n1= 1
a0 = 4
a1 = 10
b0 = 2
b1 = 3

R = 5000
est_n = np.ones(R)
# generate data
np.random.seed(123)

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
        # calculate propensity score
        f[i] = 1-1/(1+np.exp(n0+n1*L[i]))

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

    Tn = sum/n
    est_n[r] = Tn

# calc bias
bias_n = np.mean(est_n) - (a1 -a0)
print(f'bias {bias_n}')

# calc var
var_n = np.mean((est_n - np.mean(est_n))**2)
print(f'variance {var_n}')

# calc MSE
MSE_n = bias_n**2 + var_n
print(f'MSE {MSE_n}')
