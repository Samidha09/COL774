import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from ipywidgets import interactive
from matplotlib import cm
import pickle
import seaborn as sns
import time

np.random.seed(43)

# PART A


def sample_data():
    m = 1000000  # number of data points
    theta = np.array([[2], [1], [3]])  # given
    mu_1, sigma_1 = 3, 2
    mu_2, sigma_2 = -1, 2
    # examples : X values
    x_0 = np.ones(m)
    x_1 = np.random.normal(mu_1, sigma_1, m)
    x_2 = np.random.normal(mu_2, sigma_2, m)
    X = np.row_stack((x_2, x_1, x_0))  # making design matrix
    # examples: Y values
    sigma_3 = math.sqrt(2)
    Y = []  # np.random.normal(mu_3, sigma_3, m)
    for i in range(m):
        X_i = X.T[i]
        mu_3 = np.dot(theta.T, X_i)
        epsilon = np.random.normal(0, sigma_3)
        Y.append(mu_3+epsilon)
    Y = (np.array(Y)).reshape((len(Y), 1))
    return X, Y

# PART B

# function to calculate J(theta)


def cost_function(Y, theta, X):
    # number of examples for taking average
    m = X.shape[1]
    Y_hat = np.dot(theta.T, X)
    # computing mse loss
    J_theta = (1/(2*m)) * np.sum((Y - Y_hat.T)**2, axis=0)
    return J_theta

# function to calculate gradient


def gradient(theta, X, Y):
    # number of examples for taking average
    m = X.shape[1]
    Y_hat = np.dot(X.T, theta)
    # computing gradient of mse loss
    grad_vector = (-1/m) * np.dot(X, (Y - Y_hat))
    return grad_vector


def shuffle_data(X, Y):
    p = np.random.permutation(X.shape[1])
    return X[:, p], Y[p]


def SGD(X, Y, learning_rate, r, delta=0.001, k=1000):
    m = X.shape[1]
    X, Y = shuffle_data(X, Y)
    theta = np.zeros((X.shape[0], 1))
    num_batches = m/r
    t = 1
    timestep = 0
    max_epochs = 50000
    current_avg_cost = 0
    last_avg_cost = 0
    # for plotting purpose
    theta_tracker = [theta]
    cost_tracker = []
    while(1):
        start_idx = (t - 1)*r
        end_idx = (t*r)
        batch_x = X[:, start_idx:end_idx]  # [start_idx, end_idx)
        batch_y = Y[start_idx:end_idx, :]
        # update parameters for batch
        current_avg_cost += cost_function(batch_y, theta, batch_x)
        theta = theta - learning_rate*gradient(theta, batch_x, batch_y)
        theta_tracker.append(theta)
        t = t+1 if(t < num_batches) else 1
        if(timestep % k == 0):
            current_avg_cost /= k  # make cost average
            cost_tracker.append(current_avg_cost)
            if((abs(current_avg_cost - last_avg_cost) <= delta) or (timestep >= max_epochs)):
                break
            last_avg_cost = current_avg_cost
            current_avg_cost = 0
        timestep += 1

    return (theta, theta_tracker, timestep, cost_tracker)

# PART C


def load_test_data():
    X = np.genfromtxt('../Data/q2/q2test.csv', delimiter=',')
    # adding X_0 for intercept
    # print(X.shape)
    # to remove label of columns as it was string and giving nan value
    X_0 = np.ones(X.shape[0]-1)
    X_1 = X[1:, 0]
    X_2 = X[1:, 1]
    # extract labels
    test_labels = (X[1:, 2]).reshape((X.shape[0]-1, 1))
    # extract features
    # each example is a column in test_data
    test_data = np.row_stack((X_2, X_1))
    test_data = np.row_stack((test_data, X_0))
    return test_data, test_labels


def test(test_data, test_labels, theta, final_theta_dict):
    original_error = cost_function(test_labels, theta, test_data)
    test_error = {}
    for key in final_theta_dict:
        params = final_theta_dict[key]
        test_error[key] = cost_function(test_labels, params, test_data)
    return test_error, original_error

# PART D


def plot(parameter_list, elevation, angle, batch_size):
    # setting up the figure size
    fig = plt.figure(figsize=[12, 15])
    # setting up the axes as a 3 dimensional plot
    ax = fig.gca(projection='3d')

    # values for plotting line
    theta_2_vals = (parameter_list[:, 0, :].flatten()).tolist()
    theta_1_vals = (parameter_list[:, 1, :].flatten()).tolist()
    theta_0_vals = (parameter_list[:, 2, :].flatten()).tolist()

    # plot the surface
    ax.plot3D(theta_1_vals, theta_0_vals, theta_2_vals, 'red', marker='.')
    ax.set_title(
        'Movement of parameters at each iteration of SGD for batch size '+str(batch_size))
    ax.set_xlabel('theta_1')
    ax.set_ylabel('theta_0')
    ax.set_zlabel('theta_2')
    plt.savefig('./SGD_movemement_params_'+str(batch_size)+'.png')
    plt.clf()


def plot_params(parameter_dict):
    for key in parameter_dict.keys():
        parameter_list = np.array(parameter_dict[key])
        plot(parameter_list, 5, 15, key)

# UTILITY FUNCTIONS


def plot_cost(cost, num_iter, k, batch_size):
    x_s = []
    i = 0
    while(i <= num_iter):
        x_s.append(i)
        i += k
    plt.title("Cost vs timesteps")
    plt.xlabel("timesteps")
    plt.ylabel('Cost')
    plt.plot(x_s, cost)
    plt.savefig('./SGD_cost_vs_timestep'+str(batch_size)+'.png')
    plt.clf()


def save_pickle_files(final_theta_dict, parameter_dict, num_iter_dict, cost_dict):
    f1 = open('./parameter_dict.pickle', 'wb')
    pickle.dump(parameter_dict, f1)
    f2 = open('./final_theta_dict.pickle', 'wb')
    pickle.dump(final_theta_dict, f2)
    f3 = open('./num_iter_dict.pickle', 'wb')
    pickle.dump(num_iter_dict, f3)
    f4 = open('./cost_dict.pickle', 'wb')
    pickle.dump(cost_dict, f4)


def load_pickle_files():
    f1 = open('./parameter_dict.pickle', 'rb')
    parameter_dict = pickle.load(f1)
    f2 = open('./final_theta_dict.pickle', 'rb')
    final_theta_dict = pickle.load(f2)
    f3 = open('./num_iter_dict.pickle', 'rb')
    num_iter_dict = pickle.load(f3)
    f4 = open('./cost_dict.pickle', 'rb')
    cost_dict = pickle.load(f4)
    return (final_theta_dict, parameter_dict, num_iter_dict, cost_dict)

# Main function


def main():
    # get training dataset - part(a)
    X, Y = sample_data()

    # part(b) SGD
    theta = np.array([[2], [1], [3]])  # theta2, theta1, theta0
    learning_rate = 0.001

    # various configurations for stopping criteria of SGD
    delta = {1: 10**(-3), 100: 10**(-4), 10000: 10**(-6), 1000000: 10**(-8)}
    k = {1: 300, 100: 150, 10000: 100, 1000000: 1}
    batch_sizes = [1, 100, 10000, 1000000]

    final_theta_dict = {}  # key:batch_size, val: list of parameters
    parameter_dict = {}  # key:batch_size, val: list of list(3*1) of parameters
    num_iter_dict = {}  # key:batch_size, val: num_iterations_for_convergence
    cost_dict = {}  # key:batch_size, val: list of average costs of k batches
    time_taken = {}  # key:batch_size, val: time taken in seconds
    for batch_size in batch_sizes:
        start_time = time.time()
        final_theta_dict[batch_size],  parameter_dict[batch_size], num_iter_dict[batch_size], cost_dict[batch_size] = SGD(
            X, Y, learning_rate, batch_size, delta[batch_size], k[batch_size])
        time_taken[batch_size] = time.time() - start_time
        # plot cost vs iterations
        print("done: ", batch_size)
        plot_cost(cost_dict[batch_size],
                  num_iter_dict[batch_size],  k[batch_size], batch_size)

    save_pickle_files(final_theta_dict, parameter_dict,
                      num_iter_dict, cost_dict)
#     final_theta_dict, parameter_dict, num_iter_dict, cost_dict = load_pickle_files()

#     print final_parameters
    print('\nFinal Parameters\n')
    for batch_size in batch_sizes:
        print("batch size: ", batch_size, " -> params: ",
              final_theta_dict[batch_size])
#     print number of iterations taken for convergence
    print('\nNumber of iterations taken for convergence\n')
    for batch_size in batch_sizes:
        print("batch size: ", batch_size,
              " -> num_iter: ", num_iter_dict[batch_size])
#     print time taken for convergence
    print('\nTime taken(in sec) for convergence\n')
    for batch_size in batch_sizes:
        print("batch size: ", batch_size,
              " -> time_taken: ", time_taken[batch_size])

    # print difference between learned parameters and original parameters
    print('\nL2 norm of original parameters and learned parameters\n')
    for batch_size in batch_sizes:
        print("batch size: ", batch_size, " -> norm: ",
              np.linalg.norm(theta-final_theta_dict[batch_size], 2))

    # part(c) Test
    test_data, test_labels = load_test_data()
    (test_error, original_error) = test(test_data, test_labels, theta,
                                        final_theta_dict)  # dictionary, key:batch_size, value:mse error on test data
    # print error on test data
    print('\nError values on test data\n')
    print("Error with original parameters: ", original_error)
    for batch_size in batch_sizes:
        print("batch size: ", batch_size,
              " -> error : ", test_error[batch_size])

    # part(d) Plot
    plot_params(parameter_dict)
    # plots are saved in current directory and can be viewed from there


main()
