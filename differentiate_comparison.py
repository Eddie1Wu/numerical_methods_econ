
###################################### ReadMe #########################################

# I first used SymPy to derive the analytical expression of the score function,
# by differentiating with respect to beta, delta and alpha. 
# Then I analytically derived the expression of E(H(.)) and sigma_hat.

# The relevant functions and their analytical forms can be found in the functions section.
# The codes for printing results can be found in the task section.

# To show all the results, simply run this entire script will do.
# All outputs are printed to the console.

#######################################################################################


import numpy as np
import pandas as pd
import numdifftools as nd
import sympy as sym

import warnings
warnings.filterwarnings('ignore')



###################################### Functions #########################################
def objective_fun(param):
	"""
	Returns the objective function in the maximization problem
	Args:
	param - (beta, delta, alpha)
	"""
	c_t = 4000 / ( ( param[0]**data[0] * param[1]**data[1] * data[2] )**(1/(1-param[2])) + data[2] )
	
	return np.sum(-(data[3] - c_t)**2)



def score(param):
	"""
	Computes the score function analytically. The analytical expressions are derived using Sympy.
	Args:
	param - (beta, delta, alpha)
	"""

	d_beta = -8000*data[0]*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) \
				*(data[3] - 4000/(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))) \
				/(param[0]*(1 - param[2])*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	d_delta = -8000*data[1]*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) \
				*(data[3] - 4000/(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))) \
				/(param[1]*(1 - param[2])*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	d_alpha = -8000*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) *(data[3] - 4000 \
				/(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))) *np.log(param[0]**data[0]*param[1]**data[1]*data[2]) \
				/((1 - param[2])**2*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)

	return np.sum(np.array([d_beta, d_delta, d_alpha]), axis=1)



def E_hessian(param):
	"""
	Computes the expected hessian analytically. The analytical expressions of hessian
	is given by (1/N)(summation( (-2)(dg/dtheta)(dg/dtheta)' )), as shown on white board in class.
	This expression converges in probability to E(H(.)).
	Args:
	param - (beta, delta, alpha)
	"""
	
	dg_dbeta = -4000*data[0]*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) \
				/(param[0]*(1 - param[2])*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	dg_ddelta = -4000*data[1]*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) \
				/(param[1]*(1 - param[2])*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	dg_dalpha = -4000*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2]))*np.log(param[0]**data[0]*param[1]**data[1]*data[2]) \
				/((1 - param[2])**2*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	
	h11 = np.sum(dg_dbeta**2)
	h12 = np.sum(dg_dbeta * dg_ddelta)
	h13= np.sum(dg_dbeta * dg_dalpha)
	h22 = np.sum(dg_ddelta**2)
	h23 = np.sum(dg_ddelta * dg_dalpha)
	h33 = np.sum(dg_dalpha**2)

	return -2*np.array([ [h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])



def sigma_hat(param):
	"""
	Computes sigma_hat in the asymptotic variance expression analytically from the score function.
	Args:
	param - (beta, delta, alpha)
	"""

	d_beta = -8000*data[0]*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) \
				*(data[3] - 4000/(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))) \
				/(param[0]*(1 - param[2])*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	d_delta = -8000*data[1]*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) \
				*(data[3] - 4000/(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))) \
				/(param[1]*(1 - param[2])*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)
	d_alpha = -8000*(param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])) *(data[3] - 4000 \
				/(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))) *np.log(param[0]**data[0]*param[1]**data[1]*data[2]) \
				/((1 - param[2])**2*(data[2] + (param[0]**data[0]*param[1]**data[1]*data[2])**(1/(1 - param[2])))**2)

	s11 = np.sum(d_beta**2)
	s12 = np.sum(d_beta * d_delta)
	s13= np.sum(d_beta * d_alpha)
	s22 = np.sum(d_delta**2)
	s23 = np.sum(d_delta * d_alpha)
	s33 = np.sum(d_alpha**2)

	return (1/N)*np.array([ [s11, s12, s13], [s12, s22, s23], [s13, s23, s33] ])



def avar(param, hessian):
	"""
	Computes the asymptotic variance matrix.
	Args:
	param - (beta, delta, alpha)
	hessian - a 3x3 hessian matrix
	"""

	sigma = sigma_hat(param)
	bread = np.linalg.inv((1/N)*hessian)

	return bread@sigma@bread



def con_interval(param, hessian):
	"""
	Computes the confidence interval of estimators.
	Args:
	param - (beta, delta, alpha)
	hessian - a 3x3 hessian matrix
	"""

	asymptotic_var = avar(param, hessian)
	out = np.full((3,2), np.inf)
	
	for idx in range(len(param)):
		out[idx] = [param[idx]-1.96*np.sqrt(asymptotic_var[idx,idx]), param[idx]+1.96*np.sqrt(asymptotic_var[idx,idx])]

	return out



###################################### Load dataset ######################################

df = pd.read_stata("../ctb_sample.dta")

# Store data in a np array
data = df[["t0", "k", "p", "c_t"]].copy().values.T
N = data.shape[1] # Number of observations
print(N)

# Optimal param values that we estimated
param_optimal = np.array([0.9765236458797972, 0.9947526922944294, 0.6907403790821879])
# Estimates of (beta,delta,alpha) =  [0.9765236458797972, 0.9947526922944294, 0.6907403790821879]




###################################### Task ##############################################

### Derive the desired functions analytically by sympy
# Declare symbols
c, B, t, D, k, P, a = sym.symbols("c, B, t, D, k, P, a")

# Get the derivative of g(.)
g = 4000/( (B**t * D**k * P)**(1/(1-a)) + P)
print(g.diff(B))
print(g.diff(D))
print(g.diff(a))

# Get the derivative of m(.)
m = - (c - ( 4000/( (B**t * D**k * P)**(1/(1-a)) + P) ) )**2
print(m.diff(B))
print(m.diff(D))
print(m.diff(a))
print("@@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ \n")

# Print the score function
score_fun = score(param_optimal)
print("The score function is")
print(score_fun)



### Obtain Hessian in 3 ways
# 1. Directly apply 2nd order numerical differentiation to m(.)
Hfun = nd.Hessian(objective_fun)
print("Hessian from 2nd order numerical differentiation")
print(Hfun(param_optimal))

# 2. Apply 1st order numerical differentiation to the score function
Gfun = nd.Gradient(score)
print("Hessian from 1nd order numerical differentiation on score function")
print(Gfun(param_optimal))

# 3. Analytically derived E(H(.))
print("Hessian from analytical derivation of E(H(.))")
print(E_hessian(param_optimal))



### Find the asymptotic variance and 95% confidence interval
# Show all three methods
hessian = [Hfun(param_optimal), Gfun(param_optimal), E_hessian(param_optimal)] # Indicate the choice of Hessian matrix here
method = ["1", "2", "3"]

for i in range(len(hessian)):
	hes = hessian[i]
	# Compute the asymptotic variance matrix
	a_var = avar(param_optimal, hes)
	print(f"Using method {method[i]}, the asymptotic variance is")
	print(a_var)
	# Compute the confidence interval
	conf_interval = con_interval(param_optimal, hes)
	print(f"Using method {method[i]}, the confidence interval for (beta, delta, alpha) is")
	print(conf_interval)







