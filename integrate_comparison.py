
###################################### ReadMe ############################################

# Simply run the entire script which takes about 2 hours despite being highly vectorized. 
# Results will be saved to .out files in the same directoryas this script. 
# The .out files can be opened as .txt files. 

# Details of the .out files:
# - 20gh_gridsearch.out: Gauss-Hermite with n=20 for integration, gridsearch for minimization
# - 50gh_gridsearch.out: Gauss-Hermite with n=50 for integration, gridsearch for minimization
# - 100mc_gridsearch.out: MC with 100 draws for integration, gridsearch for minimization
# - 1000mc_gridsearch.out: MC with 1000 draws for integration, gridsearch for minimization
# - gh_nm.out: Gauss-Hermite for integration, Nelder-Mead for minimization
# - mc_nm.out: MC for integration, Nelder-Mead for minimization

# - If there are 4 columns in the file, they are 
#   (beta, delta, alpha, optimal value).
# - If there are 5 columns in the file, they are 
#   (choice of n for integration method, beta, delta, alpha, optimal value),
#   where n is the number of draws in Monte Carlo and 
#   n is the quadrature order in Gauss-Hermite

##########################################################################################


import numpy as np
import pandas as pd
import scipy.optimize as optim
from scipy.special import roots_hermite

import matplotlib.pyplot as plt


###################################### Functions #########################################

def future_var(x, alpha):
	"""
	Args:
	x - the epsilon in \tilde{P}
	alpha - it is just to carry this parameter to this function, don't specify any value.
	"""

	return ((4000-(p+x)*ct) / ct) ** (alpha-1) * (p+x)



def objective_fun(param, method, *args):
	"""
	Args:
	param - a numpy array of beta, delta, alpha
	method - the method for integration
	*args - positional arguments for the method
	"""

	beta, delta, alpha = param

	return np.sum( (beta**t0 * delta**k * method(future_var, *args, alpha) - 1)**2 )



def crude_mc(f, num, mean, sd, alpha):
	"""
	Args:
	f - the function to be integrated
	num - number of draws
	mean - mean of the normal distribution
	sd - standard deviation of the normal distribution
	alpha - it is just to carry this parameter to this function, don't specify any value.
	"""

	np.random.seed(seed)
	x = np.random.normal(mean, sd, (num,1))
	y = f(x, alpha)
	
	return np.sum(y, axis=0) / len(y)



def gauss_hermite(f, n, mean, sd, alpha):
	"""
	f - the function to be integrated
	n - quadrature order
	mean - mean of the normal distribution
	sd - standard deviation of the normal distribution
	alpha - it is just to carry this parameter to this function, don't specify any value.
	"""

	points, weights = roots_hermite(n)
	points = points.reshape(points.size, 1) # Reshape for broadcasting
	weights = weights.reshape(weights.size, 1) # Reshape for broadcasting
	out = (1/np.sqrt(np.pi)) * np.sum( weights * future_var(np.sqrt(2)*sd*points + mean, alpha), axis=0 )

	return out



def make_grid(beta, delta, alpha):
	"""
	beta - (start, stop, step)
	delta - (start, stop, step)
	alpha - (start, stop, step)
	"""

	b_grid = np.arange(*beta)
	d_grid = np.arange(*delta)
	a_grid = np.arange(*alpha)

	bb, dd, aa = np.meshgrid(b_grid, d_grid, a_grid)
	bb = bb.flatten()
	dd = dd.flatten()
	aa = aa.flatten()

	return bb, dd, aa



def grid_search(fun, method, grid, n, *args):
	"""
	Args:
	fun - the function to be minimized
	method - the method for integration
	grid - the grid returned by make_grid()
	n - the number of n required for the method
	*args - positional arguments for the method
	"""

	bb, dd, aa = grid
	out = np.full((bb.size,4), np.inf)

	for i in range(bb.size):
		
		param0 = np.array([bb[i], dd[i], aa[i]])
		val = fun(param0, method, n, *args)
		
		out[i,0:3] = param0
		out[i,3] = val
		
		if i%100 == 0:
			print(f"Finished {i+1} out of {bb.size} iterations")
	
	out = out[out[:,-1].argsort()]

	np.savetxt(f"{n}{name}"+'_gridsearch.out', out, delimiter=',')




###################################### Load dataset ######################################

df = pd.read_stata("../ctb_sample_new.dta")
print(df.shape)
print(df.columns)

# Store data in numpy arrays
data = df[["t0", "k", "p", "c_t_new"]].copy().values.T
t0 = data[0]
k = data[1]
p = data[2]
ct = data[3]



###################################### Tasks #############################################

# Set hyperparameters
seed = 1
mean = 0
sd = np.sqrt(0.0001)

# Set up grid
beta_range = (0.7,0.999,0.01)
delta_range = (0.8,0.999,0.01)
alpha_range = (0.5,0.799,0.01)
grid = make_grid(beta_range, delta_range, alpha_range)




######## Grid search with Monte Carlo ########
n_list = [100,1000]

for n in n_list:
	name = "mc"
	grid_search(objective_fun, crude_mc, grid, n, mean, sd)




######## Nelder Mead with Monte Carlo ########
n_list = [100, 1000, 2000, 5000, 10000]
out = np.full((len(n_list), 5), np.inf)

for i in range(len(n_list)):
	param0 = np.array([0.9, 0.9, 0.6])
	result = optim.minimize(objective_fun, param0, (crude_mc, n_list[i], mean, sd), method="Nelder-Mead")
	out[i, 0] = n_list[i]
	out[i, 1:4] = result.x
	out[i, 4] = result.fun

	print(f"Finished {i+1} out of {len(n_list)} values of n.")

np.savetxt('mc_nm.out', out, delimiter=',')




######## Grid Search with Gauss-Hermite ########
n_list = [20, 50]

for n in n_list:
	name = "gh"
	grid_search(objective_fun, gauss_hermite, grid, n, mean, sd)




######## Nelder Mead with Gauss-Hermite ########
n_list = [20, 50]
out = np.full((len(n_list), 5), np.inf)

for i in range(len(n_list)):
	param0 = np.array([0.9, 0.9, 0.6])
	result = optim.minimize(objective_fun, param0, (gauss_hermite, n_list[i], mean, sd), method="Nelder-Mead")
	out[i, 0] = n_list[i]
	out[i, 1:4] = result.x
	out[i, 4] = result.fun

	print(f"Finished {i+1} out of {len(n_list)} values of n.")

np.savetxt('gh_nm.out', out, delimiter=',') 








