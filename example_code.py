from rFS import rfs 
import numpy as np 

n = 100
p = 10
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
model = rfs(Xtrn, Ytrn, Xval=Xval, Yval=Yval)

print("rFS.B: ", model.B)
print("rfs.B0: ", model.B0)
print("rFS.regressors: ", model.regressors)

#With K-fold CV 
model = rfs(Xtrn, Ytrn)

print("rFS.B: ", model.B)
print("rfs.B0: ", model.B0)
print("rFS.regressors: ", model.regressors)

#Prediction and error methods 
print("Prediction: ", model.predict(Xtst))
print("Test RMSE: ", model.RMSE(Xtst, Ytst))