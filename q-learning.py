import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import time

"""
Example matrix representation
r = np.array([[0,1,1,1,0,0,0,0,0,0],
				[0,0,0,0,1,0,0,0,0,0],
				[0,1,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,1,1,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,1,1,0],
				[0,0,0,0,0,0,0,0,0,1],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,1,0,0],
				[0,0,0,0,0,0,0,0,0,0]])
Example dictionary representation
g = {1:[2, 3, 4], 2:[5], 3:[2], 4:[6, 7], 5:[], 6:[8, 9], 7:[10], 8:[], 9:[8], 10:[]}
"""


class QAgent():
    # Initialize alpha, gamma, rewards, and Q-values
	def __init__(self, alpha, gamma, graph, Q=None, consequence=None, random=False, risks=None):
		self.gamma = gamma  
		self.alpha = alpha
		self.rewards = self.create_rewards_matrix(graph)
		self.size = len(self.rewards[0])
		self.consequence = consequence
		if random:
			self.generate_random_rewards(self.rewards)
		if consequence != None:
			if risks is None:
				self.risks = self.generate_risks(self.rewards)
			else:
				self.risks = risks
		else:
			self.risks = risks
		self.actions = list(range(self.size))
		if Q is None:
			self.Q = np.zeros((self.size, self.size))
		else:
			self.Q = Q

	def create_rewards_matrix(self, graph):
		matrix = [[0 for _ in range(len(graph))] for _ in range(len(graph))]
		for node, neighbors in graph.items():
			for neighbor in neighbors:
				matrix[node - 1][neighbor - 1] = 1
		return matrix

	def generate_random_rewards(self, r):
		for x in range(self.size):
			for y in range(self.size):
				if r[x][y] == 1:
					r[x][y] = round(random.uniform(0, 10))

	def generate_risks(self, r):
		risks = np.array(np.zeros([self.size,self.size]))
		for x in range(self.size):
			for y in range(self.size):
				if r[x][y] > 0:
					risks[x][y] = round(random.uniform(0, 1), 2)
		return risks
	
	def training(self, start_location, end_location, iterations):
		rewards_new = np.copy(self.rewards)
		rewards_new[end_location - 1, end_location - 1] = 50
		
		for i in range(iterations):
			current_state = np.random.randint(0,self.size) 
			playable_actions = []
			for j in range(self.size):
				if rewards_new[current_state][j] > 0:
					playable_actions.append(j)
			if playable_actions:
				next_state = np.random.choice(playable_actions)
				reward = rewards_new[current_state][next_state]
				if self.consequence == None:
					risk_reward = reward
				else:
					risk = self.risks[current_state][next_state]
					if self.consequence == "neg":
						risk_reward = (1 - risk) * reward
					elif self.consequence == "neg-frac":
						risk_reward = (1 - risk) * reward + risk * (-reward * FRACTION)

				TD = risk_reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[current_state][next_state]
				self.Q[current_state][next_state] += self.alpha * TD
		return self.get_optimal_route(start_location, end_location, self.Q)
		
	def get_optimal_route(self, start_location, end_location, Q):
		start = start_location - 1
		end = end_location - 1
		current_state = start
		route = [str(current_state + 1)]
		next_state = 0
		count = 0
		while(next_state != end) and count < MAX_RUNS:
			next_state = np.argmax(Q[current_state, ])
			route.append(str(next_state + 1))
			current_state = next_state
			count += 1
		success = True if end_location == int(route[-1]) else False
		# self.print_helper(start_location, end_location, route, success)
		return success
	
	def print_helper(self, start, end, r, success):
		if success:
			# print("Rewards Table:")
			# self.print_matrix(self.rewards)
			# Risk table only applicable when there are consequences
			# if self.consequence != None:
			# 	print("Risk Table:")
			# 	self.print_matrix(self.risks)
			print("Q-Table:")
			self.print_matrix(self.Q)
			print("Optimal path from", start, "to", str(end) + ":")
			print(" -> ".join(r))
		else:
			print("FAILED")
			print("Could not find path from", start, "to", end)
			print(" -> ".join(r))
		print()

	# Helper function for displaying matrix
	def print_matrix(self, matrix):
		for row in matrix:
			print("[", end="  ")
			for val in row:
				rounded = round(val, 1)
				spaces = " " * (5 - len(str(rounded)))
				print(rounded, end=spaces)
			print("]")
		print()


MAX_RUNS = 32
FRACTION = 0.5

"""
Possible Consequences:
"neg": no reward
"neg-frac": negative of a fraction of the reward
None (default): No consequence
"""
alpha = 0.1 # Learning rate
gamma = 0.7 # Discount factor

g = {1:[1, 2, 10, 11], 2: [2, 3, 4, 5, 6, 7], 3: [3, 8, 9], 4: [4, 8, 9], 5: [5, 8, 9], 6: [6, 8, 9], 7: [7, 8, 9], 8: [8], 9: [9, 13], 10: [10], 11: [11, 12, 13], 12: [12], 13: [13, 14], 14: [14, 15, 16, 17, 18, 19], 15: [15, 20], 16: [16, 20], 17: [17, 21], 18: [18, 21], 19: [19, 21], 20: [20, 22], 21: [21, 22], 22: [22, 23, 24, 25], 23: [23], 24: [24, 27], 25: [25, 26, 27], 26: [26], 27: [27, 28], 28: [27, 28, 29], 29: [28, 29, 30, 32], 30: [29, 30, 31], 31: [30, 31], 32: [32]}

def success_rate(alpha, gamma, graph, con, start, end, reps):
	data = []
	times = []
	for r in reps:
		count = 0
		start_time = time.time()
		for i in range(100):
			agent = QAgent(alpha, gamma, graph, consequence=con)
			success = agent.training(start, end, r)
			if success:
				count += 1
		end_time = time.time()
		t = end_time - start_time
		print(f"Finished {r} iterations in {t:.2f} seconds")
		data.append([r/1000, count])
		times.append([r/1000, t])
	return data, times

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def calculate_r_squared(y_observed, y_predicted):
    ss_total = np.sum((y_observed - np.mean(y_observed))**2)
    ss_residual = np.sum((y_observed - y_predicted)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def plot_list(data):
	x_values = [point[0] for point in data]
	y_values = [point[1] for point in data]
	initial_guesses = [max(y_values), np.median(x_values), 1, min(y_values)]
	params, covariance = curve_fit(sigmoid, x_values, y_values, p0=initial_guesses)
	y_fit = sigmoid(x_values, *params)

	r_squared = calculate_r_squared(y_values, y_fit)
	plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)

	plt.scatter(x_values, y_values, label='Data Points')
	plt.plot(x_values, y_fit, color='red', label='Sigmoid Fit')
	return plt

runs = list(range(1000, 35001, 1000))

# normal = success_rate(alpha, gamma, g, None, 1, 31, runs)
# normal_rate = plot_list(normal[0])
# normal_rate.ylim(0, 100)
# normal_rate.xlabel('Iterations (Thousands)')
# normal_rate.ylabel('Success Rate (%)')
# normal_rate.title('Success Rate Based on Iterations')
# normal_rate.show()

# normal_time = plot_list(normal[1])
# normal_time.ylim(0, 20)
# normal_time.xlabel('Iterations (Thousands)')
# normal_time.ylabel('Time (seconds)')
# normal_time.title('Algorithm Runtime Based on Iterations')
# normal_time.show()


# neg = success_rate(alpha, gamma, g, "neg", 1, 31, runs)
# neg_rate = plot_list(neg[0])
# neg_rate.ylim(0, 100)
# neg_rate.xlabel('Iterations (Thousands)')
# neg_rate.ylabel('Success Rate (%)')
# neg_rate.title('Success Rate Based on Iterations (Neg Consequences)')
# neg_rate.show()

# neg_time = plot_list(neg[1])
# neg_time.ylim(0, 60)
# neg_time.xlabel('Iterations (Thousands)')
# neg_time.ylabel('Time (seconds)')
# neg_time.title('Algorithm Runtime Based on Iterations (Neg Consequences)')
# neg_time.show()


neg_frac = success_rate(alpha, gamma, g, "neg-frac", 1, 31, runs)
negfrac_rate = plot_list(neg_frac[0])
negfrac_rate.ylim(0, 100)
negfrac_rate.xlabel('Iterations (Thousands)')
negfrac_rate.ylabel('Success Rate (%)')
negfrac_rate.title('Success Rate Based on Iterations (-0.5 Consequences)')
negfrac_rate.show()

negfrac_time = plot_list(neg_frac[1])
negfrac_time.ylim(0, 60)
negfrac_time.xlabel('Iterations (Thousands)')
negfrac_time.ylabel('Time (seconds)')
negfrac_time.title('Algorithm Runtime Based on Iterations (-0.5 Consequences)')
negfrac_time.show()
