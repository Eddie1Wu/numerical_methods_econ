
import matplotlib.pyplot as plt


class Malthusian():
	"""
	The vanilla Malthusian model in our macro class

	"""

	def __init__(self, A, X, L, alpha, gamma, rho):
		self.A, self.X, self.alpha, self.gamma, self.rho = A, X, alpha, gamma, rho
		self.L = L
		self.Y = (self.A * self.X)**self.alpha * self.L**(1-self.alpha)
		self.y = self.Y/self.L
		self.period = 0
		self.converged = False


	def L_next(self):
		# Returns next period L
		return self.gamma/self.rho * (self.A * self.X)**self.alpha * self.L**(1-self.alpha)


	def y_next(self):
		# Return next period y
		return (self.rho/self.gamma)**self.alpha * self.y**(1-self.alpha)


	def L_ss(self):
		# Return the steady state L
		return (self.gamma/self.rho)**(1/self.alpha)*self.A*self.X


	def y_ss(self):
		# Return the steady state y
		return self.rho / self.gamma


	def update(self):
		# Update L, Y and y
		self.L = self.L_next()
		self.y = self.y_next()
		self.Y = self.L * self.Y
		self.period += 1


	def check_convergence(self, threshold):
		# Returns True if converged, False if not converged
		if abs(self.L - self.L_ss()) < threshold:
			self.converged = True





############# Test the class above #############

# Define params
params = (2, 2, 1, 0.5, 0.5, 0.4)   # A, X, L, alpha, gamma, rho
threshold = 0.001


# Initiate models
model1 = Malthusian(*params)
model2 = Malthusian(*params)
model2.L = 15  # Make model 2 a high population economy


# Plot convergence
fig, ax = plt.subplots()


for model in model1, model2:
	# Plot dynamics for each model
    dynamics = []
    lb = 'Dynamics from initial population {}'.format(model.L)

    while not model.converged:
    	dynamics.append(model.L)
    	model.update()
    	model.check_convergence(threshold)

    ax.plot(dynamics, 'o-', lw=2, alpha=0.6, label=lb)

ax.plot([model1.L_ss()]*max(model1.period + 3, model2.period + 5), 'k-', label='steady state') # Plot the steady state value of L

ax.legend(loc='lower right')
plt.show()





############# Bullet point 4 #############
# Define params
params = (1, 2, 1, 0.5, 0.3, 0.6)   # A, X, L, alpha, gamma, rho
threshold = 0.001


# Initiate models
model = Malthusian(*params)

# Plot convergence
fig, ax = plt.subplots()

# Plot dynamics
dynamics = []
lb = 'Dynamics from initial population {}'.format(model.L)

while not model.converged:
	dynamics.append(model.L)
	model.update()
	model.check_convergence(threshold)

ax.plot(dynamics, 'o-', lw=2, alpha=0.6, label=lb)

ax.plot([model.L_ss()]*model.period, 'k-', label='steady state') # Plot the steady state value of L

ax.legend(loc='lower right')
plt.show()





