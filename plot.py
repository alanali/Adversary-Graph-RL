from qlearning import QAgent
from game import ManipulationAgent as MAgent
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class PlotData():
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
			if count == 0:
				rewards.append([r/1000, 0])
			else:
				rewards.append([r/1000, reward/count])
		return data, rewards, times

	# Shortest path reward vs path when accounting for risks
	def reward_comp(alpha, gamma, graph, con, start, end, reps):
		r_rewards, rewards = [], []
		start_time = time.time()
		for r in reps:
			r_count, r_reward = 0, 0
			count, reward = 0, 0
			for i in range(20):
				r_agent = QAgent(alpha, gamma, graph, consequence=con, random=True)
				risks = r_agent.risks
				reward_matrix = r_agent.rewards
				r_ret = r_agent.training(start, end, r)

				agent = QAgent(alpha, gamma, graph, consequence=1, rewards=reward_matrix)
				ret = agent.training(start, end, r)
				route = ret[2]
				add = agent.route_reward(route, risks, con)
				if ret[0]:
					count += 1
					reward += add
				if r_ret[0]:
					r_count += 1
					r_reward += r_ret[1]
			end_time = time.time()
			t = end_time - start_time
			print(f"Finished {r} iterations in {t:.2f} seconds ({r/t:.2f}/sec)")
			if r_count == 0:
				r_rewards.append([r/1000, 0])
			else:
				r_rewards.append([r/1000, r_reward/r_count])
			if count == 0:
				rewards.append([r/1000, 0])
			else:
				rewards.append([r/1000, reward/count])
		return r_rewards, rewards

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
		b = ([-1, min(x_values), 1, -1], [max(y_values), max(x_values), max(y_values)+1, 5])
		params, covariance = curve_fit(PlotData.sigmoid, x_values, y_values, p0=initial_guesses, bounds=b, maxfev=5000)

		y_fit = PlotData.sigmoid(x_values, *params)
		equation = f'y = {params[0]:.2f} / (1 + exp(-{params[2]:.2f} * (x - {params[1]:.2f}))) + {params[3]:.2f}'
		plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
		r_squared = PlotData.calculate_r_squared(y_values, y_fit)
		plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
		plt.scatter(x_values, y_values, label='Data Points')
		plt.plot(x_values, y_fit, color='red', label='Sigmoid Fit')
		plt.xlabel('Iterations (Thousands)')
		return plt

	def plot_two_sig(data1, data2):
		x_values1 = [point[0] for point in data1]
		y_values1 = [point[1] for point in data1]
		initial_guesses1 = [max(y_values1), np.median(x_values1), 1, min(y_values1)]
		b1 = ([-1, min(x_values1), 1, -1], [max(y_values1), max(x_values1), max(y_values1) + 1, 5])
		params1, covariance1 = curve_fit(PlotData.sigmoid, x_values1, y_values1, p0=initial_guesses1, bounds=b1, maxfev=5000)

		y_fit1 = PlotData.sigmoid(x_values1, *params1)
		equation1 = f'y = {params1[0]:.2f} / (1 + exp(-{params1[2]:.2f} * (x - {params1[1]:.2f}))) + {params1[3]:.2f}'
		
		x_values2 = [point[0] for point in data2]
		y_values2 = [point[1] for point in data2]
		initial_guesses2 = [max(y_values2), np.median(x_values2), 1, min(y_values2)]
		b2 = ([-1, min(x_values2), 1, -1], [max(y_values2), max(x_values2), max(y_values2) + 1, 5])
		params2, covariance2 = curve_fit(PlotData.sigmoid, x_values2, y_values2, p0=initial_guesses2, bounds=b2, maxfev=5000)

		y_fit2 = PlotData.sigmoid(x_values2, *params2)
		equation2 = f'y = {params2[0]:.2f} / (1 + exp(-{params2[2]:.2f} * (x - {params2[1]:.2f}))) + {params2[3]:.2f}'

		plt.annotate(equation1, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
		r_squared1 = PlotData.calculate_r_squared(y_values1, y_fit1)
		plt.annotate(f'R-squared 1: {r_squared1:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
		plt.scatter(x_values1, y_values1, label='Regular')
		plt.plot(x_values1, y_fit1, color='blue', label='Regular')

		plt.annotate(equation2, xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=10)
		r_squared2 = PlotData.calculate_r_squared(y_values2, y_fit2)
		plt.annotate(f'R-squared 2: {r_squared2:.2f}', xy=(0.5, 0.8), xycoords='axes fraction', ha='center', fontsize=10)
		plt.scatter(x_values2, y_values2, label='Modified Risks')
		plt.plot(x_values2, y_fit2, color='orange', label='Modified Risks')

		plt.xlabel('Iterations (Thousands)')
		plt.legend(loc='lower right')
		return plt

	def plot_two_lin(data1, data2):
		x_values1 = [point[0] for point in data1]
		y_values1 = [point[1] for point in data1]
		slope1, intercept1 = np.polyfit(x_values1, y_values1, 1)
		regression_line1 = [slope1 * x + intercept1 for x in x_values1]
		equation1 = f'y = {slope1:.2f}x + {intercept1:.2f}'

		x_values2 = [point[0] for point in data2]
		y_values2 = [point[1] for point in data2]
		slope2, intercept2 = np.polyfit(x_values2, y_values2, 1)
		regression_line2 = [slope2 * x + intercept2 for x in x_values2]
		equation2 = f'y = {slope2:.2f}x + {intercept2:.2f}'

		plt.annotate(equation1, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
		r_squared1 = PlotData.calculate_r_squared(y_values1, regression_line1)
		plt.annotate(f'R-squared 1: {r_squared1:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
		plt.scatter(x_values1, y_values1, label='Data Points 1')
		plt.plot(x_values1, regression_line1, color='blue')

		plt.annotate(equation2, xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=10)
		r_squared2 = PlotData.calculate_r_squared(y_values2, regression_line2)
		plt.annotate(f'R-squared 2: {r_squared2:.2f}', xy=(0.5, 0.8), xycoords='axes fraction', ha='center', fontsize=10)
		plt.scatter(x_values2, y_values2, label='Data Points 2')
		plt.plot(x_values2, regression_line2, color='orange', label='Risk Modification')

		plt.xlabel('Iterations (Thousands)')
		plt.legend(loc='lower right')
		return plt


	# Linear Plot
	def plot_lin(data):
		x_values = [point[0] for point in data]
		y_values = [point[1] for point in data]
		slope, intercept = np.polyfit(x_values, y_values, 1)
		regression_line = [slope * x + intercept for x in x_values]
		equation = f'y = {slope:.2f}x + {intercept:.2f}'
		plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
		r_squared = PlotData.calculate_r_squared(y_values, regression_line)
		plt.annotate(f'R-squared: {r_squared:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
		plt.scatter(x_values, y_values, label='Data Points')
		plt.plot(x_values, regression_line, color='red', label='Linear Regression')
		plt.xlabel('Iterations (Thousands)')
		return plt

	# Exponential Plot
	def plot_exp(data):
		x_values = [point[0] for point in data]
		y_values = [point[1] for point in data]
		params, covariance = curve_fit(exponential, x_values, y_values, p0=(max(y_values) - min(y_values), 0, min(y_values)), maxfev = 5000)
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

	def plot_helper(consequence, runs, graph, alpha, gamma, show=True, save=True):
		run = PlotData.success_rate(alpha, gamma, graph, consequence, 1, 31, runs)
		rates = run[0]
		rewards = run[1]
		times = run[2]
		time_bound = 35
		reward_upper = 50

		# Rate Plot
		plot = PlotData.plot_sig(rates)
		plot.ylim(0, 100)
		plot.ylabel('Success Rate (%)')
		plot.title(f'Success Rate Based on Iterations ({consequence}r)')
		if save:
			plot.savefig(f'./Graphs/{consequence}_rate.png', dpi=1200)
		if show:
			plot.show()
		plot.clf()

		# Reward Plot
		plot = PlotData.plot_sig(rewards)
		plot.ylim(0, reward_upper)
		plot.ylabel('Average Total Reward')
		plot.title(f'Average Reward Based on Iterations ({consequence}r)')
		if save:
			plot.savefig(f'./Graphs/{consequence}_reward.png', dpi=1200)
		if show:
			plot.show()
		plot.clf()

		# Time Plot
		plot = PlotData.plot_lin(times)
		plot.ylim(0, time_bound)
		plot.ylabel('Time (Seconds)')
		plot.title(f'Algorithm Runtime Based on Iterations ({consequence}r)')
		if save:
			plot.savefig(f'./Graphs/{consequence}_time.png', dpi=1200)
		if show:
			plot.show()
		plot.clf()

	def comp_helper(consequence, runs, graph, alpha, gamma, show=True, save=True):
		run = PlotData.reward_comp(alpha, gamma, graph, consequence, 1, 31, runs)
		r_rewards = np.array(run[0])
		rewards = np.array(run[1])
		plt.scatter(r_rewards[:, 0], r_rewards[:, 1], label='Risk')
		plt.scatter(rewards[:, 0], rewards[:, 1], label='No Risk')
		# Linear regression for r_rewards
		r_slope, r_intercept = np.polyfit(r_rewards[:, 0], r_rewards[:, 1], 1)
		r_regression_line = [r_slope * x + r_intercept for x in r_rewards[:, 0]]
		plt.plot(r_rewards[:, 0], r_regression_line, color='blue', linestyle='--', label='Linear Regression (Risk)')
		# Linear regression for rewards
		slope, intercept = np.polyfit(rewards[:, 0], rewards[:, 1], 1)
		regression_line = [slope * x + intercept for x in rewards[:, 0]]
		plt.plot(rewards[:, 0], regression_line, color='orange', linestyle='--', label='Linear Regression (No Risk)')
		plt.ylabel('Average Reward')
		plt.xlabel('Iterations (Thousands)')
		plt.title(f'Average Reward Based on Iterations ({consequence}r)')
		plt.ylim(0, 30)
		plt.legend()
		if save:
			plt.savefig(f'./Graphs/{consequence}_comp.png', dpi=1200)
		if show:
			plt.show()
		plt.clf()

	def modified_helper(consequence, runs, graph, alpha, gamma, show=True, save=True):
		normal = []
		modified = []
		normal_s = []
		modified_s = []
		for r in runs:
			start_time = time.time()
			normal_r = 0
			modified_r = 0
			normal_count = 0
			modified_count = 0
			for _ in range(20):
				# No modifications
				initial = QAgent(alpha, gamma, graph, consequence=consequence, random=True)
				rewards = initial.rewards
				risks = initial.risks
				test = initial.training(1, 31, r)
				normal_r += test[1]
				if test[0]:
					normal_count += 1

				# Modified risks
				new_risks = MAgent.manipulate(rewards, risks, initial.consequence)
				new = QAgent(alpha, gamma, graph, consequence=consequence, risks=new_risks, rewards=rewards)
				mod = new.training(1, 31, r)
				modified_r += mod[1]
				if mod[0]:
					modified_count += 1
			if normal_r >= 0:
				normal.append([r/1000, normal_r/20])
			if modified_r >= 0:
				modified.append([r/1000, modified_r/20])
			normal_s.append([r/1000, normal_count * 5])
			modified_s.append([r/1000, modified_count * 5])
			end_time = time.time()
			t = end_time - start_time
			print(f"Finished {r} iterations in {t:.2f} seconds ({r/t:.2f}/sec)")
		
		plt = PlotData.plot_two_sig(normal, modified)

		plt.ylabel('Average Reward')
		plt.title(f'Average Reward After Risk Modification ({consequence}r)')
		plt.ylim(0, 35)
		if save:
			plt.savefig(f'./Graphs/{consequence}_reward_mod.png', dpi=1200)
		if show:
			plt.show()
		plt.clf()

		plt = PlotData.plot_two_sig(normal_s, modified_s)
		plt.ylabel('Success Rate (%)')
		plt.title(f'Success Rate with Risk Modification ({consequence}r)')
		plt.ylim(0, 100)
		if save:
			plt.savefig(f'./Graphs/{consequence}_rate_mod.png', dpi=1200)
		if show:
			plt.show()
		plt.clf()

	def print_time(time):
		mins = time // 60
		secs = time - mins * 60
		print(f"Total Time: {mins:.0f}:{secs:02.0f}")
		print()

	@staticmethod
	def generate_graphs(graph, alpha, gamma, runs, show, save):
		cons = np.arange(1, -1.1, -0.5).tolist()
		for c in cons:
			print(f"CONSEQUENCE: {c}r")
			start_time = time.time()
			PlotData.plot_helper(c, runs, graph, alpha, gamma, show, save)
			end_time = time.time()
			t = end_time - start_time
			PlotData.print_time(t)

	@staticmethod
	def compare_graphs(graph, alpha, gamma, runs, show, save):
		cons = np.arange(0.5, -1.1, -0.5).tolist()
		for c in cons:
			print(f"CONSEQUENCE: {c}r")
			start_time = time.time()
			PlotData.comp_helper(c, runs, graph, alpha, gamma, show, save)
			end_time = time.time()
			t = end_time - start_time
			PlotData.print_time(t)

	@staticmethod
	def modified_graphs(graph, alpha, gamma, runs, show, save):
		cons = np.arange(0.5, -1.1, -0.5).tolist()
		for c in cons:
			print(f"CONSEQUENCE: {c}r")
			start_time = time.time()
			PlotData.modified_helper(c, runs, graph, alpha, gamma, show, save)
			end_time = time.time()
			t = end_time - start_time
			PlotData.print_time(t)