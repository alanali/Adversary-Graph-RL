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
	y_observed = np.array(y_observed)
	y_predicted = np.array(y_predicted)
	ss_total = np.sum((y_observed - np.mean(y_observed))**2)
	ss_residual = np.sum((y_observed - y_predicted)**2)
	r_squared = 1 - (ss_residual / ss_total)
	return r_squared

# Sigmoid Plot
def plot_sig(data):
	x_values = [point[0] for point in data]
	y_values = [point[1] for point in data]
	initial_guesses = [max(y_values), np.median(x_values), 1, min(y_values)]
	b = ([0, min(x_values), 0, 0], [100, max(x_values), 10, 5])
	params, covariance = curve_fit(sigmoid, x_values, y_values, p0=initial_guesses, bounds=b)
	y_fit = sigmoid(x_values, *params)
	equation = f'y = {params[0]:.2f} / (1 + exp(-{params[2]:.2f} * (x - {params[1]:.2f}))) + {params[3]:.2f}'
	plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
	r_squared = calculate_r_squared(y_values, y_fit)
	plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
	plt.scatter(x_values, y_values, label='Data Points')
	plt.plot(x_values, y_fit, color='red', label='Sigmoid Fit')
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
	return plt


runs = list(range(1000, 10001, 500))
many_runs = list(range(1000, 35001, 1000))

normal = success_rate(alpha, gamma, g, 0, 1, 31, runs)

normal_rate = plot_sig(normal[0])
normal_rate.ylim(0, 100)
normal_rate.xlabel('Iterations (Thousands)')
normal_rate.ylabel('Success Rate (%)')
normal_rate.title('Success Rate Based on Iterations')
normal_rate.savefig('./Graphs/zero_rate.png', dpi=1200)
normal_rate.show()

normal_time = plot_lin(normal[1])
normal_time.ylim(0, 20)
normal_time.xlabel('Iterations (Thousands)')
normal_time.ylabel('Time (Seconds)')
normal_time.title('Algorithm Runtime Based on Iterations')
normal_time.savefig('./Graphs/zero_time.png', dpi=1200)
normal_time.show()


neg = success_rate(alpha, gamma, g, -1, 1, 31, runs)

neg_rate = plot_sig(neg[0])
neg_rate.ylim(0, 100)
neg_rate.xlabel('Iterations (Thousands)')
neg_rate.ylabel('Success Rate (%)')
neg_rate.title('Success Rate Based on Iterations (-r)')
neg_rate.savefig('./Graphs/neg_rate.png', dpi=1200)
neg_rate.show()

neg_time = plot_lin(neg[1])
neg_time.ylim(0, 20)
neg_time.xlabel('Iterations (Thousands)')
neg_time.ylabel('Time (Seconds)')
neg_time.title('Algorithm Runtime Based on Iterations (-r)')
neg_time.savefig('./Graphs/neg_time.png', dpi=1200)
neg_time.show()


neg_frac = success_rate(alpha, gamma, g, -0.5, 1, 31, runs)

negfrac_rate = plot_sig(neg_frac[0])
negfrac_rate.ylim(0, 100)
negfrac_rate.xlabel('Iterations (Thousands)')
negfrac_rate.ylabel('Success Rate (%)')
negfrac_rate.title('Success Rate Based on Iterations (-0.5r)')
negfrac_rate.savefig('./Graphs/neg_half_rate.png', dpi=1200)
negfrac_rate.show()

negfrac_time = plot_lin(neg_frac[1])
negfrac_time.ylim(0, 20)
negfrac_time.xlabel('Iterations (Thousands)')
negfrac_time.ylabel('Time (Seconds)')
negfrac_time.title('Algorithm Runtime Based on Iterations (-0.5r)')
negfrac_time.savefig('./Graphs/neg_half_time.png', dpi=1200)
negfrac_time.show()
