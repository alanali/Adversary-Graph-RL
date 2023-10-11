import numpy as np
import random

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

MAX_RUNS = 10

class QAgent():
    # Initialize alpha, gamma, rewards, and Q-values
	def __init__(self, alpha, gamma, graph, consequence=None, random=False, risks=None):
		self.gamma = gamma  
		self.alpha = alpha
		self.rewards = self.create_rewards_matrix(graph)
		self.size = len(self.rewards[0])
		self.consequence = consequence
		if random:
			self.generate_random_rewards(self.rewards)
		if consequence != None:
			if risks != None:
				self.risks = risks
			else:
				self.risks = self.generate_risks(self.rewards)
		else:
			self.risks = risks
		self.actions = list(range(self.size))
		self.Q = np.random.rand(self.size, self.size) * 0

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
		start_location -= 1
		end_location -= 1
		rewards_new[end_location, end_location] = 50
		
		for i in range(iterations):
			current_state = np.random.randint(0,self.size) 
			playable_actions = []
			for j in range(self.size):
				if rewards_new[current_state][j] > 0:
					playable_actions.append(j)
			if playable_actions:
				next_state = np.random.choice(playable_actions)
				if self.consequence == "neg":
					risk_reward = (1 - self.risks[current_state][next_state]) * rewards_new[current_state][next_state]
				elif self.consequence == None:
					risk_reward = rewards_new[current_state][next_state]

				TD = risk_reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[current_state][next_state]
				self.Q[current_state][next_state] += self.alpha * TD
		self.get_optimal_route(start_location, end_location, self.Q)
		
	def get_optimal_route(self, start_location, end_location, Q):
		current_state = start_location
		route = [str(current_state + 1)]
		next_state = 0
		count = 0
		while(next_state != end_location) and count < MAX_RUNS:
			next_state = np.argmax(Q[current_state, ])
			route.append(str(next_state + 1))
			current_state = next_state
			count += 1
		success = True if (end_location + 1) == int(route[-1]) else False
		self.print_helper(start_location + 1, end_location + 1, route, success)
	
	def print_helper(self, start, end, r, success):
		if success:
			print("Rewards Table:")
			self.print_matrix(self.rewards)
			# Risk table only applicable when there are consequences
			if self.consequence != None:
				print("Risk Table:")
				self.print_matrix(self.risks)
			print("Q-Table:")
			self.print_matrix(self.Q)
			print("Optimal path from", start, "to", str(end) + ":")
			print(" -> ".join(r))
		else:
			print("FAILED")
			print("Could not find path from", start, "to", end)
			print(" -> ".join(r))

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


"""
Possible Consequences:
"neg": no reward
None (default): No consequence
"""
alpha = 0.1 # Learning rate
gamma = 0.7 # Discount factor

g = {1:[1, 2, 10, 11], 2: [2, 3, 4, 5, 6, 7], 3: [3, 8, 9], 4: [4, 8, 9], 5: [5, 8, 9], 6: [6, 8, 9], 7: [7, 8, 9], 8: [8], 
	9: [9, 13], 10: [10], 11: [11, 12, 13], 12: [12], 13: [13, 14], 14: [14, 15, 16, 17, 18, 19], 15: [15, 20], 16: [16, 20], 17: [17, 21], 18: [18, 21], 19: [19, 21], 20: [20, 22], 21: [21, 22], 22: [22, 23, 24, 25], 23: [23], 24: [24, 27], 25: [25, 26, 27], 26: [26], 27: [27, 28], 28: [27, 28, 29], 29: [28, 29, 30, 32], 30: [29, 30, 31], 31: [30, 31], 32: [32]}

print("NO CONSEQUENCE")
qagent = QAgent(alpha, gamma, g)
qagent.training(1, 22, 100000)

print("\n\nNEG CONSEQUENCE")
qagent = QAgent(alpha, gamma, g, consequence="neg")
qagent.training(1, 22, 100000)