###################################### Read Me ###########################################
#
# My pipeline for finding the optimal parameter values that minimize the objective function:
#
# 1. Do a coarse grid search over the parameter space to collect the set of parameter
#    estimates and objective function values.
#
# 2. Using the set in step 1 as initial values, run derivative-based methods. 
#
# 3. For robustness check, I also try other initial values, a different way of computing
#    the objective function, and other derivative-based methods. 
#
# 4. Using the set in step 1 as initial values, run derivative-free methods including 
#    a finer grid search.
#
# 5. Validate my result by plotting the objective function against each of the parameters 
#    over an interval around the optimal value, while keeping the other parameters fixed
#    at the optimal value.
#
#
# The optimal (beta, delta, alpha) I found are:
#
# (0.97652207, 0.99475226, 0.69072394)
#
# and the minimum objective function value is: 
#
# 10648197149.108326
#
#
# Please note that it takes several minutes to run the entire script in one go.
# But each section is self-contained. You could comment out the unwanted sections and
# only run the wanted sections.
#
##########################################################################################


import pandas as pd
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt


###################################### Functions #########################################

def square_residual(x, param):
	"""
	Calculates the square residual for one observation
	Args:
	x - data vector with entries t0, k, P, c_t
	param - beta, delta, alpha
	"""
	beta_delta_P = ((param[0]**x[0]) * (param[1]**x[1]) * x[2]) ** (1/(param[2]-1))

	return ( x[3] -  4000*beta_delta_P / (1 + x[2]*beta_delta_P) )**2



def objective_func_vec(param):
	"""
	Calculates the objective function with vectorization
	Args:
	param - beta, delta, alpha
	"""
	y = np.apply_along_axis(square_residual, 1, data_array, param=param).sum()

	return y



def objective_func_loop(param):
	"""
	Calculates the objective function by looping over rows
	Args:
	param - beta, delta, alpha
	"""
	y = 0

	for i in range(data_array.shape[0]):

		beta_delta_P = ( (param[0]**data_array[i,0]) * (param[1]**data_array[i,1]) * data_array[i,2] )**(1/(param[2]-1))

		y += ( data_array[i,3] -  4000*beta_delta_P / (1 + data_array[i,2]*beta_delta_P) )**2

	return y



def make_grid(b_bound, d_bound, a_bound):
	"""
	Makes a grid for beta, alpha and delta. Returns 3 vectors whose i-th entries are the coordinates of the i-th point in the grid
	Args:
	b_bound - a tuple (lower, upper, n) for beta, where n is the number of points
	d_bound - a tuple (lower, upper, n) for delta
	a_bound - a tuple (lower, upper, n) for alpha
	"""
	b_grid = np.linspace(b_bound[0],b_bound[1],b_bound[2])
	d_grid = np.linspace(d_bound[0],d_bound[1],d_bound[2])
	a_grid = np.linspace(a_bound[0],a_bound[1],a_bound[2])

	beta,delta,alpha = np.meshgrid(b_grid, d_grid, a_grid)
	return beta.flatten(), delta.flatten(), alpha.flatten()



def grid_search(func, beta, delta, alpha, return_all=False):
	"""
	Implements grid search
	Args:
	func - objective function that takes a (3,) array of params
	beta - beta vector returned by make_grid()
	delta - delta vector returned by make_grid()
	alpha - alpha vector returned by make_grid()
	"""
	out = []

	for i in range(len(beta)):
		param = np.array([beta[i], delta[i], alpha[i]])
		out.append(func(param))

		if i%5 == 0:
			print(f"Completed {i} out of {len(beta)} values in the grid.")

	out = np.array(out)
	idx = np.argmin(out)

	if return_all:
		return beta, delta, alpha, out
	else:
		return beta[idx], delta[idx], alpha[idx], out[idx]



def param_plot(grid_bound, optimal_param, interest_param):
	"""
	Produces a plot where the parameter of interest varies while the other parameters are fixed
	Args:
	grid_bound - a tuple (lower, upper, n) for the parameter of interest, where n is the number of points
	optimal_param - a tuple (beta, delta, alpha) which is optimal
	interest_param - 0 for beta, 1 for delta, 2 for alpha
	"""

	grid = np.linspace(grid_bound[0], grid_bound[1] , grid_bound[2])
	vals = []

	for g in grid:

		if interest_param == 0:
			param = np.array([g, optimal_param[1], optimal_param[2]])
		elif interest_param == 1:
			param = np.array([optimal_param[0], g, optimal_param[2]])
		elif interest_param == 2:
			param = np.array([optimal_param[0], optimal_param[1], g])

		vals.append(objective_func_vec(param))

	print(grid[np.argmin(vals)])
	plt.plot(grid, vals)
	plt.axvline(x = optimal_param[interest_param], color = 'b', label = 'Optimal')
	plt.legend()
	plt.show()





###################################### Load dataset ######################################

df = pd.read_stata("../ctb_sample.dta")
print(df.columns)
print(df.head(5))

# Store data in a np array
data_array = df[["t0", "k", "p", "c_t"]].copy().values
print(data_array.shape)





###################################### Task ##############################################


######## First do a coarse grid search to pin down initial values for other search methods 

# Create a grid of intial values
beta_grid, delta_grid, alpha_grid = make_grid([0.6,1,8], [0.6,1,8], [0.3,0.8,8])

# Find the optimal initial values
beta, delta, alpha, val = grid_search(objective_func_vec, beta_grid, delta_grid, alpha_grid, return_all=True) 
out = np.array([beta, delta, alpha, val]).T
out = out[out[:, -1].argsort()]
print(out[0:3, :]) # This shows the top 3 sets of initial values that achieve minimum
print(out.shape)
print(f"The minimum is achieved at beta {out[0,0]}, delta {out[0,1]}, alpha {out[0,2]}, value = {out[0,3]}")
# Should print The minimum is achieved at beta 0.7142857142857143, delta 1.0, alpha 0.44285714285714284, value = 11978205266.778347





######## Run derivative-based methods using the top 3 sets of initial values found above 

# Run BFGS using the top 3 sets of initial values found above
param_list = [ np.array([0.71, 0.99, 0.44]), np.array([0.71, 0.99, 0.37]), np.array([0.66, 0.99, 0.37]) ]
for param0 in param_list:
	out = optim.minimize(objective_func_vec, param0, method="BFGS")
	print(out.x)   # Should print sth similar to [0.97652207 0.99475226 0.69072394]
	print(out.fun)   # Should print sth similar to 10648197149.117043


# For robustness check, try other initial values
param_list = [np.array([0.8, 0.9, 0.4]), np.array([0.7, 0.9, 0.4]), np.array([0.7, 0.85, 0.44]), np.array([0.7, 0.9, 0.6]), np.array([0.7, 0.9, 0.7])]
for param0 in param_list:
	out = optim.minimize(objective_func_vec, param0, method="BFGS")
	print(out.x) 
	print(out.fun)  # These starting values should all print the same outputs


# For robustness check, try the objective function that is computed by a loop
param0 = np.array([0.71, 0.99, 0.44])
out = optim.minimize(objective_func_loop, param0, method="BFGS")
print(out.x)   # Should print [0.9765219  0.99475226 0.69072422]
print(out.fun)   # Should print 10648197149.113686


# For robustness check, try another derivative-based method
param0 = np.array([0.71, 0.99, 0.44])
out = optim.minimize(objective_func_loop, param0, method="L-BFGS-B")
print(out.x)   # Should print [0.97652157 0.99475228 0.69072446]
print(out.fun)   # Should print 10648197149.108326





######## Run derivative-free methods using the initial values found above

# Run Nelder-Mead using the top 3 sets of initial values found above
param_list = [ np.array([0.71, 0.99, 0.44]), np.array([0.71, 0.99, 0.37]), np.array([0.66, 0.99, 0.37]) ]
for param0 in param_list:
	out = optim.minimize(objective_func_vec, param0, method="Nelder-Mead")
	print(out.x)   # Should print sth similar to [0.97652143 0.99475228 0.69072425]
	print(out.fun)   # Should print sth similar to 10648197149.105358


# Try the objective function that is calculated by looping
param0 = np.array([0.71, 0.99, 0.44])
out = optim.minimize(objective_func_loop, param0, method="Nelder-Mead")
print(out.x)   # Should print [0.9765215  0.99475227 0.69072424]
print(out.fun)   # Should print 10648197149.105026


# Run fine grid search using the optimal values found above
beta_grid, delta_grid, alpha_grid = make_grid([0.97,0.98,8], [0.99,1,8], [0.688,0.693,8]) # Create a grid of intial values

beta, delta, alpha, val = grid_search(objective_func_vec, beta_grid, delta_grid, alpha_grid) # Find the optimal initial values

print(f"The minimum is achieved at beta {beta}, delta {delta}, alpha {alpha}, value = {val}")
# Should print The minimum is achieved at beta 0.98, delta 0.9942857142857143, alpha 0.688, value = 10666470000.639133





######## Validate my result found above by plotting

optimal_param = np.array([0.97652207, 0.99475226, 0.69072394])

# Vary beta only
param_plot([0.95,0.99,100], optimal_param, interest_param=0)

# Vary delta only
param_plot([0.99,1,100], optimal_param, interest_param=1)

# Vary alpha only
param_plot([0.68,0.70,100], optimal_param, interest_param=2)








