
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt


# Function
f = lambda x: 3*x**4 - 5*x**3 + 2*x**2

# Plotting
x = np.linspace(-0.5, 1.5, 100)
y = f(x)
plt.plot(x,y)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.show()




################################ Question 1 #################################

# List of starting values
x0 = [-0.25, 0, 0.25, 0.5, 0.75, 1]
x_newton = []
x_nelder = []

for x in x0:
	### Q1. Newton's method
	x_newton.append(float(optim.minimize(f, x, method='BFGS').x))
	
	### Q2. Nelder-Mead method
	x_nelder.append(float(optim.minimize(f, x, method='Nelder-Mead').x))

print("The optimal values in Question 1 are:")
print(x_newton)
print(x_nelder)





################################# Question 2 #################################

### Grid search
grid = np.linspace(-1,1, 1000)
y = f(grid)
minimum = grid[np.argmin(y)]
print("The minimum value from grid search is:")
print(minimum)














