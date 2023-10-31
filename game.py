from qlearning import QAgent
import numpy as np
from scipy.optimize import minimize
import random

# Before red makes a move, blue is able to manipulate risk values of red's possible moves to lower red's overall reward value. Red then makes a move, oblivious to the manipulation.

class ManipulationAgent():
	@staticmethod
	def manipulate(rewards, risks, consequence):
		new_risks = []  # Create a new list to store modified risk values
		for r, rk in zip(rewards, risks):
			non_zero_indices = [i for i, value in enumerate(r) if value != 0]
			if non_zero_indices:
				b = [(0, 1)] * len(non_zero_indices)
				result = minimize(ManipulationAgent.calculate_reward, rk[non_zero_indices],
									args=([r[i] for i in non_zero_indices], consequence),
									constraints={'type': 'eq', 'fun': lambda x: sum(x) - sum(rk)},
									bounds=b)  # Create a copy of the original risks
				modified_risks = rk.copy()
				for i, index in enumerate(non_zero_indices):
					modified_risks[index] = round(result.x[i], 2)
				new_risks.append(modified_risks)
			else:
				new_risks.append(rk)
		return new_risks

	@staticmethod
	def calculate_reward(risks, rewards, consequence):
		r = []
		for i in range(len(rewards)):
			if rewards[i] != 0:
				r.append(rewards[i] * (1 - risks[i]) + rewards[i] * risks[i] * consequence)
		return -min(r)
