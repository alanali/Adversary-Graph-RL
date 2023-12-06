from qlearning import QAgent
import numpy as np
from scipy.optimize import minimize
import random

# Before red makes a move, blue is able to manipulate risk values of red's possible moves to lower red's overall reward value. Red then makes a move, oblivious to the manipulation.

class ManipulationAgent():
	# Use the SLSQP algorithm to manipulate adversary's risk values
	@staticmethod
	def manipulate(rewards, risks, consequence):
		# Create list to store modified risk values
		new_risks = []
		for r, rk in zip(rewards, risks):
			# Get nonzero reward indexes
			non_zero_indices = [i for i, value in enumerate(r) if value != 0]
			if non_zero_indices:
				# Define bounds for optimization
				b = [(0, 1)] * len(non_zero_indices)
				# Use optimization to calculate modified risks that minimize the calculated reward
				result = minimize(ManipulationAgent.calculate_reward, rk[non_zero_indices],
									args=([r[i] for i in non_zero_indices], consequence),
									constraints={'type': 'eq', 'fun': lambda x: sum(x) - sum(rk)},
									bounds=b)
				print("Optimization method:", result.message)

				# Create a copy of the original risks
				modified_risks = rk.copy()
				# Update the modified risks with the calculated values
				for i, index in enumerate(non_zero_indices):
					modified_risks[index] = round(result.x[i], 2)
				new_risks.append(modified_risks)
			else:
				# If all rewards are zero, return the original risks
				new_risks.append(rk)
		return new_risks

	@staticmethod
	def calculate_reward(risks, rewards, consequence):
		r = []
		# Calculate the overall reward for each action based on risks, rewards, and consequence
		for i in range(len(rewards)):
			if rewards[i] != 0:
				r.append(rewards[i] * (1 - risks[i]) + rewards[i] * risks[i] * consequence)
		# Return the minimum calculated reward
		return min(r)
