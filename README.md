# rFS: Regularized Forward Stepwise Selection (version 0.1)
Scalable python tool for building sparse linear models for regression and classification. 

## Overview 

Many quantities of interest (Y) can be modeled using linear functions of X, as follows: 

![The Linear Model](linmodel.jpg=20x100)

Where *n* is the number of data points and *p* is the number of variables to choose from. Our task is to solve for the values of the coefficients (beta) of these variables. 

This algorithm builds an approximate solution to the following problem regularized, cardinality-constrained regression/classification problem: 

![Regularized Best Subset Selection](regbestsubset.jpg=100x20)

*k* is an integer that constrains the size of the model selected. 

rFS works by first building an initial solution using forward stepwise selection. Then, the variables that are active at each step are regularized using either a lasso-like or ridge-like penalty (as implmented in the R package [glmnet](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html). Forward stepwise selection works by including the variable in every step that minimizes the squared error loss, the solution path can be computed very efficiently. The regularization step of rFS is also very quick making the algorithm very scalable. 

## Installation & Requirements 

### R-dependency 
For now, rFS acts as a wrapper combining elements from Tibshirani et al.'s `bestsubset` package written in R. Consequently, the user will have to install: 
- [R](https://www.r-project.org/) (version 3.4 or newer)
- [bestsubset](https://github.com/ryantibs/best-subset/)

### Python 
The user will need to install the [Python](https://www.python.org/downloads/) (version 3.6 or newer).

Then, the rFS package can be installed using [pip](https://pip.pypa.io/en/stable/), running the following code from the command line: 

```bash 
pip install rFS
``` 
The following dependencies will also be automatically installed: 
- [numpy](https://numpy.org/)
- [rpy2](https://rpy2.github.io/doc/latest/html/index.html] - interface to R in python)
## Useage 

```python 
from rFS import rfs 
import numpy as np 

n = 100
p = 1000
k = 5
true_variables = np.sort(np.random.choice([i for i in range(p)], size=k, replace=False))
true_coefficients = np.zeros(p)
true_coefficients[true_variables] = 1

# Predictors
X = np.random.rand(n*3, p)
Xtrn = X[0:n, :]
Xval = X[n:2*n, :]
Xtst = X[2*n:3*n, :]
# Response
Ytrn = Xtrn@true_coefficients + .5*np.random.randn(n)
Yval = Xval@true_coefficients + .5*np.random.randn(n)
Ytst = Xtst@true_coefficients + .5*np.random.randn(n)		

print("Btrue: ", true_coefficients)
print("Regressors: ", true_variables)

#With Validation set 
#Create rfs model object 
model = rfs(Xtrn, Ytrn, Xval=Xval, Yval=Yval)

#Coefficients of chosen variables 
print("rFS.B: ", model.B)
#Intercept term 
print("rfs.B0: ", model.B0)
#List of chosen variables 
print("rFS.regressors: ", model.regressors)

#Prediction and error methods of rfs object  
#1D array to predict Y at Xtst 
print("Prediction: ", model.predict(Xtst))
#Scalar root mean squared error of model prediction at Xtst compared to actual value at Ytst 
print("Test RMSE: ", model.RMSE(Xtst, Ytst))

```

## Arugments for `rfs` class initialization

### Mandatory 

- `X, Y`: Training data for the regressors (2D array of dimension (*n*, *p*)) and response (1D array of dimension (*n*,)). First and second positional arguments, respectively, creating the `rfs` object. 

### Optional  

Must be supplied explicitly, in any order past the first two positions. 

- `Xval, Yval`: 2D and 1D arrays, respectively, for data that should be used for cross-validation. If not specified, the provided training data is used in a K-fold cross validation procedure. 
- `nfolds`: integer value for the number of folds (K) that should be used in cross validation. Default value is 10. If both `Xval, Yval` and `nfolds` is specified, algorithm will default to using explicit validation set instead of folds. 
- `reg`: specifies with regularization penalty to use, either `"L1"` for lasso-like, or `"L2"` for ridge-like. Default is `"L1"`.
- `maxSteps`: The maximum number of steps that should be used to build stepwise solution. Default value is 50. Algorithm will pick the minimum of `maxSteps`, *n*, & *p*. 

### Methods for `rfs` class

- `rfs.predict(X)`: Predict response at data points `X`
- `rfs.rmse(X, Y)`: Root mean squared error between model prediction at `X` and actual data at `Y`