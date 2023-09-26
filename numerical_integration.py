

#################################### ReadMe #######################################

# Simply run this script and all outputs will be printed to the console.
# Note that for Task 2, my laptop does not have enough RAM to store arrays with 
# (10^6)^2 elements or more, as such I was unable to do "n = 10^k for k=1,...,7".
# As such, I used "n=2^k for k=2,4,6,...14" instead.

# The entire script will require a few minutes to run.

# When running this code on a more powerful computer, simply change the list of n 
# in my code from base 2 to base 10 will produce the results as desired.

###################################################################################



import time
import numpy as np
import scipy.integrate as integ

import matplotlib.pyplot as plt



#################################### Functions ####################################

f1 = lambda x: np.exp(x)

f2 = lambda x, y: np.exp(x+y)


def midpoint_1d(f, lower, upper, num):
	"""
	Args:
	f - the function mapping from R to R
	lower - the lower limit of the domain to integrate
	upper - the upper limit of the domain to integrate
	num - the number of rectangles
	"""

	x = np.linspace(lower, upper, num+1)
	midpoint = (x[1:]+x[:-1])/2
	
	return np.sum( f(midpoint) * (x[-1]-x[0])/midpoint.size )



def midpoint_2d(f, lower, upper, num):
	"""
	Args:
	f - the function mapping from R^2 to R
	lower - the lower limit of the domain to integrate
	upper - the upper limit of the domain to integrate
	num - the number of cuboids
	"""

	x = np.linspace(lower, upper, num+1)
	y = np.linspace(lower, upper, num+1)
	x_midpoint = (x[1:]+x[:-1])/2
	y_midpoint = (y[1:]+y[:-1])/2
	xv, yv = np.meshgrid(x_midpoint, y_midpoint)

	return np.sum( f(xv, yv) * ((x[-1]-x[0])/x_midpoint.size) * ((y[-1]-y[0])/y_midpoint.size) )



def crude_mc(f, lower, upper, num, dim=1):
	"""
	Args:
	f - the function
	lower - the lower limit of the domain to integrate
	upper - the upper limit of the domain to integrate
	num - number of draws per dimension
	dim - number of dimensions of the function
	"""

	np.random.seed(seed)
	x = np.random.uniform(lower, upper, (num**dim, dim))
	y = f(x[:,0], x[:,1]) if x.shape[1]==2 else f(x)
	y_bar = np.sum(y) / len(y)

	return y_bar * (upper - lower)**dim



def trapezoid_1d(f, lower, upper, num):
	"""
	Args:
	f - the function mapping R -> R
	lower - the lower limit of the domain to integrate
	upper - the upper limit of the domain to integrate
	num - the number of trapezoids
	"""

	x = np.linspace(lower, upper, num=num)
	out = integ.trapezoid(f(x), x)
	return out



def simpsons_1d(f, lower, upper, num):
	"""
	Args:
	f - the function mapping R -> R
	lower - the lower limit of the domain to integrate
	upper - the upper limit of the domain to integrate
	num - the number of trapezoids
	"""

	x = np.linspace(lower, upper, num=num)
	out = integ.simpson(f(x), x)
	return out



def gaussian_1d(f, lower, upper):
	"""
	Args:
	f - the function mapping R -> R
	lower - the lower limit of the domain to integrate
	upper - the upper limit of the domain to integrate
	"""
	
	out = integ.fixed_quad(f, lower, upper)
	return out



def eval_n(method, truth, target, num, *args, **kwargs):
	"""
	Args:
	method - the choice of method
	truth - the true value of integral
	target - the threshold for stopping
	num - the initial choice of n
	"""

	out = method(args[0], args[1], args[2], num)
	out = out[0] if type(out) is tuple else out
	error = np.abs(out-truth)

	try:
		if error <= target:
			while error <= target:
				num -= 1
				out = method(args[0], args[1], args[2], num)
				out = out[0] if type(out) is tuple else out
				error = np.abs(out-truth)
			return num+1
		else:
			num += 20
			return eval_n(method, truth, target, num, args[0], args[1], args[2])
	except RecursionError:
		print("Crude Monte Carlo method: error does not decrease even when n is extremely large. Search terminated.")



def eval_time(method, *args, MC=False):
	""" 
	Args:
	method - the choice of method
	"""
	start = time.time()
	if MC==True:
		out = method(args[0], args[1], args[2], args[3], args[4])
	else:
		out = method(args[0], args[1], args[2], args[3])
	dt = time.time()-start

	return dt, out




#################################### Task #########################################

# Set seed
seed = 1


########## Task 1 ##########
lower = 0
upper = 2
number = 1000

# Numerically compute f1
midpt = midpoint_1d(f1, lower, upper, number)
print(midpt)
trapd = trapezoid_1d(f1, lower, upper, number)
print(trapd)
simps = simpsons_1d(f1, lower, upper, number)
print(simps)
gauss = gaussian_1d(f1, lower, upper)
print(gauss[0])
mc = crude_mc(f1, lower, upper, number)
print(mc)


# Find the number of n needed to achieve error 10e-6
target = 10e-6
number = 100
truth = np.exp(2)-1
n_midpt = eval_n(midpoint_1d, truth, target, number, f1, lower, upper)
n_trapd = eval_n(trapezoid_1d, truth, target, number, f1, lower, upper)
n_simps = eval_n(simpsons_1d, truth, target, number, f1, lower, upper)
n_mc = eval_n(crude_mc, truth, target, number, f1, lower, upper)

n_all = [n_midpt, n_trapd, n_simps, n_mc]
print("The n that leads to error below 10e-6 for [midpoint, trapezoid, simpsons, Monte Carlo] is:")
print(n_all)


# Compare the computation time
time_list = []
time_list.append(eval_time(midpoint_1d, f1, lower, upper, n_all[0])[0]*1000)
time_list.append(eval_time(trapezoid_1d, f1, lower, upper, n_all[1])[0]*1000)
time_list.append(eval_time(simpsons_1d, f1, lower, upper, n_all[2])[0]*1000)
time_list.append(eval_time(crude_mc, f1, lower, upper, n_all[3])[0]*1000)
print("The time taken in milliseconds for [midpoint, trapezoid, simpsons, Monte Carlo]")
print(time_list) # the unit is millisecond
print("@@@@@ @@@@@ @@@@@ @@@@@ @@@@@ @@@@@\n")





########## Task 2 ##########
lower = 0
upper = 2
number = 1000

# Numerically compute the function
midpt = midpoint_2d(f2, lower, upper, number)
print(midpt)
mc = crude_mc(f2, lower, upper, number, dim=2)
print(mc)


# Try different values of n. Here I'm using 2**k for k = 2, 4, 6... 14 as mentioned in the ReadMe above
power = np.arange(2, 15, 2)
n_list = 2**power
errors = np.full((len(n_list),2), np.nan)
time_list = np.full((len(n_list),2), np.nan)
truth = ( np.exp(2)-1 )**2

for i in range(len(errors)):

	t,val = eval_time(midpoint_2d, f2, lower, upper, n_list[i], MC=False)
	errors[i,0] = np.abs(val-truth)
	time_list[i,0] = t
	
	t,val = eval_time(crude_mc, f2, lower, upper, n_list[i], 2, MC=True)
	errors[i,1] = np.abs(val-truth)
	time_list[i,1] = t

	print(f"Finished {i+1} out of {len(n_list)} elements.")



# Plotting
fig, ax = plt.subplots(2,1)

ax[0].plot(range(len(n_list)), errors[:,0], label="Midpoint method")
ax[0].plot(range(len(n_list)), errors[:,1], label="Monte Carlo")
ax[1].set_ylabel("In log")
ax[0].set_title("Errors")
ax[0].legend()

ax[1].plot(range(len(n_list)), np.log(time_list[:,0]), label="Midpoint method")
ax[1].plot(range(len(n_list)), np.log(time_list[:,1]), label="Monte Carlo")
ax[1].set_ylabel("In log(milliseconds)")
ax[1].set_title("Time taken")
ax[1].legend()

plt.show()





