
import numpy as np
from numpy import linalg as LA
from scipy.optimize import broyden1
import matplotlib.pyplot as plt




#################################### Functions #####################################

def objective_func(q):
	# Args:
	# q - a (2,) array

	foc1 = (q[0]+q[1])**(-1/alpha) - (1/alpha)*(q[0]+q[1])**((-1/alpha)-1)*q[0] - c1*q[0]
	foc2 = (q[0]+q[1])**(-1/alpha) - (1/alpha)*(q[0]+q[1])**((-1/alpha)-1)*q[1] - c2*q[1]

	return np.array([foc1, foc2])



def jacobian(q):
	# Args: 
	# q - a (2,) array

	j11 = -(1/alpha)*(q[0]+q[1])**(-1/alpha -1) - ( (-1/alpha -1)*(1/alpha)*(q[0]+q[1])**(-1/alpha -2)*q[0] + (1/alpha)*(q[0]+q[1])**(-1/alpha -1) ) - c1
	j12 = -(1/alpha)*(q[0]+q[1])**(-1/alpha -1) - ( (-1/alpha -1)*(1/alpha)*(q[0]+q[1])**(-1/alpha -2)*q[0] )
	j21 = -(1/alpha)*(q[0]+q[1])**(-1/alpha -1) - ( (-1/alpha -1)*(1/alpha)*(q[0]+q[1])**(-1/alpha -2)*q[1] )
	j22 = -(1/alpha)*(q[0]+q[1])**(-1/alpha -1) - ( (-1/alpha -1)*(1/alpha)*(q[0]+q[1])**(-1/alpha -2)*q[1] + (1/alpha)*(q[0]+q[1])**(-1/alpha -1) ) - c2

	return np.array([[j11, j12], [j21, j22]])



def newton_method(f, J, x0, epsilon=1e-6, delta=1e-5, max_iter=100):
	# Args:
	# f - the objective function, takes a (2,) array as input
	# J - the Jacobian matrix function, takes a (2,) array as input
	# x0 - a (2,) array of starting values
	# epsilon - stopping threshold for x
	# delta - stopping threshold for objective function

	x1 = x0 - LA.inv(J(x0)) @ f(x0)
	iteration = 0

	while LA.norm(x0 - x1) > epsilon*(1+LA.norm(x1)) and iteration <= max_iter:
		x0 = x1
		x1 = x0 - LA.inv(J(x0)) @ f(x0) # update x value
		iteration += 1

		if iteration == max_iter:
			break

	# Print message
	if iteration == max_iter:
		print("No convergence after {} iterations, increase max_iter.".format(iteration))	
	elif LA.norm(f(x1)) > delta:
		print("Convergence achieved in {} iterations but root NOT found.".format(iteration))
	elif LA.norm(f(x1)) <= delta:
		print("Convergence achieved in {} iterations and root found.".format(iteration))

	return x1, f(x1)



def fixed_point_iteration(f, x0, tolerance, lambda_0=0, step_size=0.99, max_iter=100):
	# args:
	# lambda_0 - the initial value of lambda_k
	# step_size - the coefficient to update lambda_k
	# tolerance - the threshold for convergence
	# max_iter - the maximum number of iterations

	error = tolerance+999
	num_iter = 0

	while any(error > tolerance) and num_iter <= max_iter:
		x0 = lambda_0*x0 + (1-lambda_0)*f(x0)
		lambda_0 = lambda_0 * step_size
		error = abs(x0 - f(x0))
		num_iter += 1

	if num_iter == max_iter:
		print("Maximum number of iterations reached, no convergence.")
	elif all(error <= tolerance):
		print("Convergence achieved in {} iterations.".format(num_iter))

	return x0



def objective_func_fixed_point(q):
	# Args:
	# q - a (2,) array

	foc1 = (q[0]+q[1])**(-1/alpha) - (1/alpha)*(q[0]+q[1])**((-1/alpha)-1)*q[0] - c1*q[0] + q[0]
	foc2 = (q[0]+q[1])**(-1/alpha) - (1/alpha)*(q[0]+q[1])**((-1/alpha)-1)*q[1] - c2*q[1] + q[1]

	return np.array([foc1, foc2])




#################################### Parameters ####################################

alpha = 1.5
c1 = 0.6
c2 = 0.8




#################################### Task 1 ########################################

### Q1
x0 = np.array([5, 5])
root, value = newton_method(objective_func, jacobian, x0)

print(root)   # Output: [0.81309599 0.67109731]
print(value)



### Q2
x0 = np.array([10, 10])
root = broyden1(objective_func, x0)
print(root)   # Output: [0.81309598 0.67109731]



### Q3
x0 = np.array([5, 5])
root = fixed_point_iteration(objective_func_fixed_point, x0, tolerance=np.full((2,), 1e-8), lambda_0=1)
print(root)   # Output: [0.813096   0.67109731]




#################################### Task 2 ########################################

alpha_grid = np.linspace(1,3, 100)
quantities = []

for alpha in alpha_grid:
	x0 = np.array([10, 10])
	root = broyden1(objective_func, x0)
	quantities.append(root)

quantities = np.array(quantities)
prices = np.sum(quantities, axis=1)


# Plotting
fig, axes = plt.subplots(2)

axes[0].plot(alpha_grid, quantities[:,0], label="q1*")
axes[0].plot(alpha_grid, quantities[:,1], label="q2*")
axes[0].set_title("Optimal quantities and alpha")
axes[0].legend()

axes[1].plot(alpha_grid, prices)
axes[1].set_title("Prices and alpha")

plt.show()






