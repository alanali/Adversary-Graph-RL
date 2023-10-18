import numpy as np
import random

MAX_RUNS = 32

class QAgent():
    # Initialize alpha, gamma, rewards, and Q-values
	def __init__(self, alpha, gamma, graph, Q=None, consequence=1, random=False, risks=None):
		self.gamma = gamma  
		self.alpha = alpha
		self.rewards = self.create_rewards_matrix(graph)
		self.size = len(self.rewards[0])
		self.consequence = consequence
		if random:
			self.generate_random_rewards()
		if consequence != 1:
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

	def generate_random_rewards(self):
		for x in range(self.size):
			for y in range(self.size):
				if self.rewards[x][y] == 1:
					self.rewards[x][y] = round(random.uniform(1, 5))

	def generate_risks(self, r):
		risks = np.array(np.zeros([self.size,self.size]))
		for x in range(self.size):
			for y in range(self.size):
				if r[x][y] > 0:
					risks[x][y] = round(random.uniform(0, 1), 2)
		return risks
	
	def training(self, start_location, end_location, iterations):
		rewards_new = np.copy(self.rewards)
		rewards_new[end_location - 1, end_location - 1] = 100
		playable_actions = [np.where(rewards_new[i] > 0)[0] for i in range(self.size)]

		for _ in range(iterations):
			current_state = np.random.randint(0, self.size) 
			if playable_actions[current_state].size > 0:
				next_state = np.random.choice(playable_actions[current_state])
				reward = rewards_new[current_state][next_state]
				if self.consequence == 1:
					risk = 0
				else:
					risk = self.risks[current_state][next_state]
				risk_reward = (1 - risk) * reward + risk * (reward * self.consequence)
				TD = risk_reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[current_state, next_state]
				self.Q[current_state, next_state] += self.alpha * TD

		return self.get_optimal_route(start_location, end_location, self.Q)
			
	def get_optimal_route(self, start_location, end_location, Q):
		start = start_location - 1
		end = end_location - 1
		current_state = start
		route = [str(current_state + 1)]
		reward = 0
		for count in range(MAX_RUNS):
			next_state = np.argmax(Q[current_state, ])
			route.append(str(next_state + 1))
			if self.consequence != 1:
				chance = round(random.uniform(0, 1), 2)
				risk = self.risks[current_state][next_state]
				if chance < risk:
					reward += self.rewards[current_state][next_state] * self.consequence
			else:
				reward += self.rewards[current_state][next_state]
			current_state = next_state

			if current_state == end:
				success = True
				break
		else:
			success = False
			reward = 0
		# self.print_helper(start_location, end_location, route, success)
		return success, reward

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

