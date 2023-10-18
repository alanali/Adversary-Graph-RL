from qlearning import QAgent
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

alpha = 0.1 # Learning rate
gamma = 0.7 # Discount factor

g = {1:[1, 2, 10, 11], 2: [2, 3, 4, 5, 6, 7], 3: [3, 8, 9], 4: [4, 8, 9], 5: [5, 8, 9], 6: [6, 8, 9], 7: [7, 8, 9], 8: [8], 9: [9, 13], 10: [10], 11: [11, 12, 13], 12: [12], 13: [13, 14], 14: [14, 15, 16, 17, 18, 19], 15: [15, 20], 16: [16, 20], 17: [17, 21], 18: [18, 21], 19: [19, 21], 20: [20, 22], 21: [21, 22], 22: [22, 23, 24, 25], 23: [23], 24: [24, 27], 25: [25, 26, 27], 26: [26], 27: [27, 28], 28: [27, 28, 29], 29: [28, 29, 30, 32], 30: [29, 30, 31], 31: [30, 31], 32: [32]}

def success_rate(alpha, gamma, graph, con, start, end, reps):
	data, times, rewards = [], [], []
	for r in reps:
		count, reward = 0, 0
		start_time = time.time()
		for i in range(100):
			agent = QAgent(alpha, gamma, graph, consequence=con, random=True)
			ret = agent.training(start, end, r)
			if ret[0]:
				count += 1
				reward += ret[1]
		end_time = time.time()
		t = end_time - start_time
		print(f"Finished {r} iterations in {t:.2f} seconds ({r/t:.2f}/sec)")
		data.append([r/1000, count])
		times.append([r/1000, t])
		if count != 0:
			rewards.append([r/1000, reward/count])
		else:
			rewards.append([r/1000, 0])
	print(rewards)
	return data, rewards, times

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def calculate_r_squared(y_actual, y_predicted):
	mean_y = np.mean(y_actual)
	total_sum_squares = sum((y - mean_y) ** 2 for y in y_actual)
	residual_sum_squares = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
	r_squared = 1 - (residual_sum_squares / total_sum_squares)
	return r_squared

# Sigmoid Plot
def plot_sig(data):
	x_values = [point[0] for point in data]
	y_values = [point[1] for point in data]
	initial_guesses = [max(y_values), np.median(x_values), 1, min(y_values)]
	b = ([0, min(x_values), 0, 0], [100, max(x_values), 10, 5])
	params, covariance = curve_fit(sigmoid, x_values, y_values, p0=initial_guesses, bounds=b, maxfev=5000)
	y_fit = sigmoid(x_values, *params)
	equation = f'y = {params[0]:.2f} / (1 + exp(-{params[2]:.2f} * (x - {params[1]:.2f}))) + {params[3]:.2f}'
	plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
	r_squared = calculate_r_squared(y_values, y_fit)
	plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
	plt.scatter(x_values, y_values, label='Data Points')
	plt.plot(x_values, y_fit, color='red', label='Sigmoid Fit')
	plt.xlabel('Iterations (Thousands)')
	return plt

# Linear Plot
def plot_lin(data):
	x_values = [point[0] for point in data]
	y_values = [point[1] for point in data]
	slope, intercept = np.polyfit(x_values, y_values, 1)
	regression_line = [slope * x + intercept for x in x_values]
	equation = f'y = {slope:.2f}x + {intercept:.2f}'
	plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
	r_squared = calculate_r_squared(y_values, regression_line)
	plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
	plt.scatter(x_values, y_values, label='Data Points')
	plt.plot(x_values, regression_line, color='red', label='Linear Regression')
	plt.xlabel('Iterations (Thousands)')
	return plt

# Exponential Plot
def plot_exp(data):
	x_values = [point[0] for point in data]
	y_values = [point[1] for point in data]
	params, covariance = curve_fit(exponential, x_values, y_values, maxfev = 2000)
	a, b, c = params
	x_fit = np.linspace(min(x_values), max(x_values), 100)
	regression_line = exponential(x_fit, a, b, c)
	r_squared = calculate_r_squared(y_values, regression_line)
	equation = f'y = {a:.2f} * e^({b:.2f}x) + {c:.2f}'
	plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
	plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
	plt.scatter(x_values, y_values, label='Data Points')
	plt.plot(x_fit, regression_line, color='red', label='Exponential Regression')
	plt.xlabel('Iterations (Thousands)')
	return plt

def plot_helper(consequence, runs, show=True, save=True):
	print(f"CONSEQUENCE: {consequence}r")
	start_time = time.time()

	run = success_rate(alpha, gamma, g, consequence, 1, 31, runs)
	rates = run[0]
	rewards = run[1]
	times = run[2]
	time_bound = 60
	reward_upper = max([r[1] for r in rewards]) + 1
	reward_lower = min([r[1] for r in rewards]) - 1

	# Rate Plot
	plot = plot_sig(rates)
	plot.ylim(0, 100)
	plot.ylabel('Success Rate (%)')
	plot.title(f'Success Rate Based on Iterations ({consequence}r)')
	if save:
		plot.savefig(f'./Graphs/{consequence}_rate.png', dpi=1200)
	if show:
		plot.show()
	plot.clf()

	# Reward Plot
	plot = plot_exp(rewards)
	plot.ylim(reward_lower, reward_upper)
	plot.ylabel('Average Total Reward')
	plot.title(f'Average Reward Based on Iterations ({consequence}r)')
	if save:
		plot.savefig(f'./Graphs/{consequence}_reward.png', dpi=1200)
	if show:
		plot.show()
	plot.clf()

	# Time Plot
	plot = plot_lin(times)
	plot.ylim(0, time_bound)
	plot.ylabel('Time (Seconds)')
	plot.title(f'Algorithm Runtime Based on Iterations ({consequence}r)')
	if save:
		plot.savefig(f'./Graphs/{consequence}_time.png', dpi=1200)
	if show:
		plot.show()
	plot.clf()

	end_time = time.time()
	t = end_time - start_time
	mins = t // 60
	secs = t - mins * 60
	print(f"Total Time: {mins:.0f}:{secs:02.0f}")
	print()


def generate_graphs():
	runs = list(range(5000, 25001, 1000))
	cons = np.arange(1, -1.1, -0.5).tolist()
	for c in cons:
		plot_helper(c, runs, show=False)

generate_graphs()