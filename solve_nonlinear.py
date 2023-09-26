
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import math
import numpy as np


# Define functions
f = lambda x: math.exp((x-2)**2) - 2 - x
g = lambda x: math.exp((x-2)**2) - 2



###################### Task 1 ######################
root = root_scalar(f, method='bisect', bracket=(0.5,1.5))
print(root)




###################### Task 2.1: Solve x = g(x) by fixed point iteration ######################
def fixed_point_iteration(f, x0, lambda_0 = 0, step_size=0.99, tolerance = 10e-4, max_iter = 100):
	# args:
	# lambda_0: the initial value of lambda_k
	# step_size: the coefficient to update lambda_k
	# tolerance: the threshold for convergence
	# max_iter: the maximum number of iterations

	error = 1
	num_iter = 0
	x_seq = []

	while (error > tolerance and num_iter <= max_iter):
		x_seq.append(x0)
		x0 = lambda_0*x0 + (1-lambda_0)*f(x0)
		lambda_0 = lambda_0 * step_size
		error = abs(x0 - f(x0))
		num_iter += 1

	return x0, x_seq, num_iter

# 2.1 Ans: this does not work. There is no convergence. We run into OverFlowError and g(x) becomes too large as x grows.
# root, _ = fixed_point_iteration(g, x0 = 0.5)
# print(root)




###################### Task 2.2: Solve x = g(x) with a different updating rule ######################
root, x_seq, num_iter = fixed_point_iteration(g, x0 = 0.5, lambda_0 = 1)
print(root)

# 2.2 Ans: this works perfectly as the updating rule prevents divergence. 
#          We obtain the same root as using the SciPy package.

# Plotting
x_grid = np.linspace(0.45,1.5,num=100)
plt.plot(x_grid, [g(x) for x in x_grid], color = "r", label = "g(x)")
plt.plot(x_grid, x_grid, color = "b", label = "45 degree line")
plt.plot(x_seq, [g(x) for x in x_seq], 'bo', label='sequence')
plt.xlabel("Number of iterations")
plt.ylabel("Value of x")
plt.legend()
plt.show()





















