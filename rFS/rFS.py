#Also try objective penalized version of r1fs/r2fs
import numpy as np 
import copy 
#Import R package and functions 
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
d = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
bestsubset = importr('bestsubset', robject_translations=d)
scale = ro.r['scale']
coef = ro.r['coef']
as_matrix = ro.r['as.matrix']
predict = ro.r['predict']

class rfs(): 

	def __init__(self, X, Y, **kwargs):
		"""
		Builds linear regression model using Regularized Forward Stepwise Selection 

		Required positional arguments: 

			X : 2D numpy array of regressor training data array with dimensions (# data points, # regressors)
			Y : numpy array of response training data array with dimension (# data points)

		Optional key word arguments: 

			Xval, Yval : Similar to X and Y; Data used to cross-validate hyper-parameters 
						(if supplied, K-fold CV is not performed)
			nfolds : Int Number of folds used for K-fold cross-validation 
			reg: String regularization penalty 
				"L1" - for lasso-like 
				"L2" - for ridge-like 
			maxSteps: Int maximum number of steps for forward selection 

		Returns: 
		self function containing several attributes - 
			
			self.B : 1D numpy array of coefficients for each regressor 
			self.regressors : 1D numpy array of regressor variables selected, elements of type Int value 1 - # regressors 

		""" 
		#Flatten input response array 
		Y = np.array(Y).flatten()

		#Verify that input data is of correct dimensions
		assert len(np.shape(X)) == 2, "Regressor matrix must be a 2D array"
		assert np.shape(Y)[0] == np.shape(X)[0], "First dimension of X must be equal to dimension of Y"

		#Check kwargs and set options
		KWARGS = list(kwargs)
		Xval = kwargs["Xval"] if "Xval" in KWARGS else False 
		Yval = kwargs["Yval"] if "Yval" in KWARGS else False 
		if (type(Xval) is bool) | (type(Yval) is bool): 
			assert Xval == Yval, "Must supply both Xval and Yval"
		nfolds = kwargs["nfolds"] if "nfolds" in KWARGS else 5 
		if nfolds > np.shape(X)[0]: nfolds = np.shape(X)[0] 
		reg = kwargs["reg"] if "reg" in KWARGS else "L2"
		maxSteps = min(np.shape(X)[0], np.shape(X)[1], kwargs["maxSteps"]) if "maxSteps" in KWARGS else min(np.shape(X)[0], np.shape(X)[1])

		#Build regression model 
		self.buildModel(X, Y, Xval, Yval, nfolds, reg, maxSteps)


	def buildModel(self, X, Y, Xval, Yval, 
		nfolds, reg, maxSteps): 
		"""
		Required positional arguments: 

			X : 2D numpy array of regressor training data array with dimensions (# data points, # regressors)
			Y : 1D response training data array with dimension (# data points)
			Xval, Yval : Similar to X and Y; Data used to cross-validate hyper-parameters 
						(if supplied, K-fold CV is not performed)
			nfolds : Int Number of folds used for K-fold cross-validation 
			reg: String regularization penalty 
				"L1" - for lasso-like 
				"L2" - for ridge-like 
			maxSteps: Int maximum number of steps for forward selection 

		Returns: 
		self function containing several attributes - 
			
			self.B : 1D numpy array of coefficients for each regressor 
			self.regressors : 1D numpy array of regressor variables selected, elements of type Int value 1 - # regressors
		"""
		#Type of regularization
		alpha = 1 if reg == "L1" else 0 

		##Build model using cross-validation 

		#If validation data provided, use it for CV; else, use K-fold CV 
		if type(Xval) is bool: 

			n = np.shape(X)[0]
			fs = round(n / nfolds)
			fold_val_error_list = []

		
			for fold in range(nfolds): 
				#Define training and validation data 
				Xval = X[min(n, fold*fs):min(n, fold*fs+fs), :]
				Yval = Y[min(n, fold*fs):min(n, fold*fs+fs)]
				Xtrn, Ytrn = X.copy(), Y.copy()
				Xtrn = np.delete(Xtrn, slice(min(n, fold*fs), min(n, fold*fs+fs)), axis=0)
				Ytrn = np.delete(Ytrn, slice(min(n, fold*fs), min(n, fold*fs+fs)), axis=0)

				#Build Model 
				FSsoln = bestsubset.fs(Xtrn, Ytrn, maxsteps=maxSteps)
				step_val_error_list = []
				for step in range(2, maxSteps+1): 
					FSactiveset = np.sort(np.nonzero(np.array(bestsubset.coef_fs(FSsoln, step)))[0])
					if 0 in FSactiveset: 
						FSactiveset = np.delete(FSactiveset, 0, axis=0)
						FSactiveset -= 1 
					RegularizedSolution = bestsubset.lasso(Xtrn[:, FSactiveset], Ytrn, alpha=alpha)
					step_val_error = [sum((Yval-np.array(as_matrix(bestsubset.predict_lasso(RegularizedSolution, Xval[:, FSactiveset]))).T[l])**2)
					for l in range(50)]
					#account for FS unregularized solution 
					step_val_error.append(sum((Yval-np.array(as_matrix(bestsubset.predict_fs(FSsoln, Xval))).T[step])**2))
					#Record step error and solution 
					step_val_error_list.append(step_val_error)

				fold_val_error_list.append(np.array(step_val_error_list))

			fold_val_error_array = np.array(fold_val_error_list)
			# print(np.array(fold_val_error_array).shape)
			# print(np.array(fold_val_error_list))
			# print(np.unravel_index(np.mean(fold_val_error_array, axis=0).argmin(), np.mean(fold_val_error_array, axis=0).shape))

			opt_model = np.unravel_index(np.mean(fold_val_error_list, axis=0).argmin(), np.mean(fold_val_error_list, axis=0).shape)
			opt_step = opt_model[0] + 2 
			opt_lambda = opt_model[1]

			FSsoln = bestsubset.fs(X, Y, maxsteps=opt_step)
			FSactiveset = np.sort(np.nonzero(np.array(bestsubset.coef_fs(FSsoln, opt_step)))[0])
			if 0 in FSactiveset: 
				FSactiveset = np.delete(FSactiveset, 0, axis=0)
				FSactiveset -= 1 
			if opt_lambda != 50: 
				RegularizedSolution = bestsubset.lasso(X[:, FSactiveset], Y, alpha=alpha)
				self.B = np.array(as_matrix(bestsubset.coef_lasso(RegularizedSolution))).T[opt_lambda][1:].flatten()
				self.B0 = np.array(as_matrix(bestsubset.coef_lasso(RegularizedSolution))).T[opt_lambda][0]
			else: 
				self.B = np.array(as_matrix(bestsubset.coef_fs(FSsoln, opt_step)))[FSactiveset+1].flatten()
				self.B0 = np.array(as_matrix(bestsubset.coef_fs(FSsoln, opt_step)))[0]

			self.regressors = np.sort(FSactiveset[np.sort(np.nonzero(self.B)[0])])
	
		#Use validation data 
		else: 

			FSsoln = bestsubset.fs(X, Y, maxsteps=maxSteps)
			step_val_error_list, step_min_sol_coef_list, FSactiveset_list = [], [], []
			for step in range(2, maxSteps+1): 
				FSactiveset = np.sort(np.nonzero(np.array(bestsubset.coef_fs(FSsoln, step)))[0])
				if 0 in FSactiveset: 
					FSactiveset = np.delete(FSactiveset, 0, axis=0)
					FSactiveset -= 1 
				FSactiveset_list.append(FSactiveset)
				RegularizedSolution = bestsubset.lasso(X[:, FSactiveset], Y, alpha=alpha)
				step_val_error = [sum((Yval-np.array(as_matrix(bestsubset.predict_lasso(RegularizedSolution, Xval[:, FSactiveset]))).T[l])**2)
				for l in range(50)]
				#account for FS unregularized solution 
				step_val_error.append(sum((Yval-np.array(as_matrix(bestsubset.predict_fs(FSsoln, Xval))).T[step])**2))
				#Record step error and solution 
				step_val_error_list.append(min(step_val_error))
				if step_val_error.index(min(step_val_error)) == 50: 
					min_soln_coef = np.array(as_matrix(bestsubset.coef_fs(FSsoln, step))) \
					[np.sort(np.nonzero(np.array(as_matrix(bestsubset.coef_fs(FSsoln, step))))[0])]
				else: 
					min_soln_coef = np.array(as_matrix(bestsubset.coef_lasso(RegularizedSolution))).T[step_val_error.index(min(step_val_error))]
				step_min_sol_coef_list.append(min_soln_coef)
				if min(step_val_error) < 1e-10: 
					break 

			self.B = step_min_sol_coef_list[step_val_error_list.index(min(step_val_error_list))][1:].flatten()
			self.B0 = step_min_sol_coef_list[step_val_error_list.index(min(step_val_error_list))][0] 
			self.regressors = np.sort(FSactiveset_list[step_val_error_list.index(min(step_val_error_list))][np.sort(np.nonzero(self.B)[0])]) 



	def predict(self, X): 
		"""
		Predicts response at input data points using built model. 
		
		Required arguments: 

			X : 2D numpy array of regressor data 

		Returns: 

			Ypred : 1D numpy array of response predicted at each point 
		"""
		#No intercept 
		if self.B0 == 0: 
			Ypred = X[:, self.regressors]@self.B
		#If intercept 
		else: 
			Ypred = X[:, self.regressors]@self.B + self.B0

		return Ypred 

	def RMSE(self, Xtest, Ytest): 
		"""
		Calculates the root mean squared error of built model on a test set of data 

		Required arguments: 

			Xtest : 2D numpy array of regressor test data 
			Ytest : numpy array of response test data 

		Returns: 

			Scalar: RMSE value 
		"""
		return (sum((Ytest - self.predict(Xtest))**2)/len(np.array(Ytest).flatten()))**(0.5)
