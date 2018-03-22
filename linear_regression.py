import numpy as np

def compute_error(b, m, points):

	total_error = 0

	for i in range(len(points)):

		x = points[i, 0]
		y_true = points[i, 1]

		y_calculated = (b*x + m)
		
		total_error += np.square(y_true - y_calculated)

	return total_error / float(len(points))


def descend(points, initial_m, initial_b, learning_rate, no_of_iterations):

	b = initial_b
	m = initial_m

	for i in range(no_of_iterations):

		b, m = gradient_step(b, m, points, learning_rate)

	return [b, m]


def gradient_step(current_m, current_b, points, learning_rate):

	gradient_b = 0
	gradient_m = 0

	n = len(points)

	for i in range(len(points)):

		x = points[i, 0]
		y = points[i, 1]

		gradient_b += - (2/n)   *   (y - ((current_m * x) + current_b))
		gradient_m =  - (2/n) * x * (y - ((current_m * x) + current_b))

	new_b = current_b - (learning_rate * gradient_b)
	new_m = current_m - (learning_rate * gradient_m)

	return[new_m, new_b]


def run():

	# Prep data

	points = np.genfromtxt('dataset.csv', delimiter=',')

	# Hyperparam setup

	initial_m = 0
	initial_b = 0

	learning_rate = 0.0001
	no_of_iterations = 1000

	# Training

	[b, m] = descend(points, initial_m, initial_b, learning_rate, no_of_iterations)
	print(b)
	print(m)
	print(compute_error(b, m, points))

run()