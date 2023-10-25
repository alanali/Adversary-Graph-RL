from qlearning import QAgent
from plot import PlotData
import numpy as np
from scipy.optimize import minimize
import random

# Before red makes a move, blue is able to manipulate risk values of red's possible moves to lower red's overall reward value. Red then makes a move, oblivious to the manipulation.

class ManipulationAgent():
	@staticmethod
	def manipulate(rewards, risks, consequence):
		final_result = None

		for r, rk in zip(rewards, risks):
			print("Rewards:", r)
			print("Risks:", rk)
			print("Sum of inital risks:", sum(rk))
			b = [(0, 1)] * len([r for r in rk if r != 0])
			if any(r_i != 0 for r_i in r):
            # Filter the indices of non-zero rewards
				non_zero_indices = [i for i, value in enumerate(r) if value != 0]

				# Only optimize the corresponding risk values for non-zero rewards
				result = minimize(ManipulationAgent.calculate_reward, rk[non_zero_indices],
									args=([r[i] for i in non_zero_indices], consequence),
									constraints={'type': 'eq', 'fun': lambda x: sum(x) - sum(rk)}, bounds=b)

				# Update the corresponding risk values in the original array
				for i, index in enumerate(non_zero_indices):
					rk[index] = round(result.x[i], 2)
				print("Modified Risks:", rk)
				print("Sum of modified risks:", sum(rk))
				print()

	@staticmethod
	def calculate_reward(risks, rewards, consequence):
		overall_reward = 0
		for i in range(len(rewards)):
			if rewards[i] != 0:  # Only calculate for non-zero rewards
				overall_reward += rewards[i] * (1 - risks[i]) + rewards[i] * risks[i] * consequence
		return -overall_reward

	@staticmethod
	def risk_constraint(risks):
		return np.sum(risks) - 1
