
import numpy as np
import matplotlib.pyplot as plt


############ Functions ############

f = lambda x: np.sin(x)


def forward_diff(f, x, h):
	return ( f(x+h) - f(x) ) / h


def central_diff(f,x,h):
	return ( f(x+h) - f(x-h) ) / (2*h)

###################################


h_list = 10**( -np.arange(1.0,21.0,1.0) )
print(h_list)
x = 0.5


# Find gradient at x using the 2 methods
forward_grad = forward_diff(f, x, h_list)
central_grad = central_diff(f, x, h_list)
true_grad = np.cos(x)


# Calculate the errors
x_axis = -np.log10(h_list)
forward_error = np.log10(np.abs(true_grad - forward_grad))
central_error = np.log10(np.abs(true_grad - central_grad))


#Plotting
fig, ax = plt.subplots()
ax.plot(x_axis, central_error, label="Central differences")
ax.plot(x_axis, forward_error, label="Forward differences")
ax.axhline(y=0, color="r")
ax.set_title('Total errors')
ax.legend()
# plt.show()
plt.savefig("total_errors.png")







