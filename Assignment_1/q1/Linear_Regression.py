''' REFERENCES
1. https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725
2. https://www.youtube.com/watch?v=xd2sZ8rXLZI
3. https://medium.com/@rohitadnaik/3d-line-plot-in-python-2fbeca99b9ba
4. https://www.google.com/search?client=safari&rls=en&q=3d+plots+matplotlib+label+axes&ie=UTF-8&oe=UTF-8
5. https://matplotlib.org/stable/gallery/animation/simple_anim.html
6. https://plotly.com/python/3d-surface-plots/
7. https://pythonmatplotlibtips.blogspot.com/2017/12/draw-3d-line-animation-using-python-matplotlib-funcanimation.html'''

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML
from ipywidgets import interactive
from matplotlib import cm
import seaborn as sns
#from numpy import genfromtxt

# Writer = animation.FFMpegWriter(fps=30, codec='libx264')  #or
# Writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)


def load_data():
    # reading data
    x_data_orig = np.genfromtxt('../Data/q1/linearX.csv', delimiter=',')
    # standardization of data
    x_1 = (x_data_orig - np.mean(x_data_orig)) / np.std(x_data_orig)
    # for intercept term
    x_0 = np.ones(len(x_1))
    # stack row wise to get x, each column now represents one example
    x = np.row_stack((x_1, x_0))
    y = np.genfromtxt('../Data/q1/linearY.csv',
                      delimiter=',').reshape(len(x_1), 1)

    return x, y

# PART (a) : LINEAR REGRESSION WITH BATCH GRADIENT DESCENT

# function to calculate J(theta)


def cost_function(Y, theta, X):
    # number of examples for taking average
    m = X.shape[1]
    Y_hat = np.dot(X.T, theta)
    # computing mse loss
    J_theta = (1/(2*m)) * np.sum((Y - Y_hat)**2, axis=0)
    return J_theta

# function to calculate gradient


def gradient(theta, X, Y):
    # number of examples for taking average
    m = X.shape[1]
    Y_hat = np.dot(X.T, theta)
    # computing gradient of mse loss
    grad_vector = (-1/m) * np.dot(X, (Y - Y_hat))
    return grad_vector

# linear regression


def linear_regression(X, Y, learning_rate=0.01, delta=0.0000000001, verbose=False):
    # initializing theta parameter with all zeros
    theta = np.random.rand(X.shape[0], 1)  # np.zeros((X.shape[0], 1)) #
    time_step = 0
    # for now keeping epochs as stopping criteria
    J_theta_t = cost_function(Y, theta, X)
    # lists to track values of theta and corresponding cost for plots
    error_tracker = [J_theta_t]
    theta_1_tracker = [theta[0][0]]
    theta_0_tracker = [theta[1][0]]

    while(1):
        if(verbose):
            print("Cost at ", time_step, ":", J_theta_t)
        theta = theta - learning_rate * gradient(theta, X, Y)
        J_theta_t_plus_1 = cost_function(Y, theta, X)
        # push in tracking lists for plots
        error_tracker.append(J_theta_t_plus_1)
        theta_1_tracker.append(theta[0][0])
        theta_0_tracker.append(theta[1][0])
        # stopping criteria
        if(abs(J_theta_t - J_theta_t_plus_1) <= delta):
            break
        # if(np.linalg.norm(gradient(theta, X, Y), ord=1, axis=None, keepdims=False) < 0.00001)
        # update J_theta_t to J_theta_t_plus_1
        J_theta_t = J_theta_t_plus_1
        time_step += 1

    return (theta, error_tracker, theta_1_tracker, theta_0_tracker)


# PART (b) : PLOT

def scatter_plot(x, y, y_hat):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.title('Linear Regression: Line Of Best Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x[0, :], y, color='#2b0559', linewidth=0.7)
    plt.plot(x[0, :], y_hat, color='#eb9e3b', linewidth=0.8)
    plt.savefig('./line.png')
    plt.show()
# PART (c) : PLOT


def f(theta_1_vals, theta_0_vals, x, y):
    m = x.shape[1]
    error = []
    for i in range(theta_1_vals.shape[0]):
        error_i = []
        for j in range(theta_1_vals.shape[1]):
            theta_1 = theta_1_vals[i][j]
            theta_0 = theta_0_vals[i][j]
            theta = np.row_stack((theta_1, theta_0))
            y_hat = np.dot(x.T, theta)
            J_theta = (1/(2*m)) * np.sum((y - y_hat)**2, axis=0)
            error_i.append(np.squeeze(J_theta, axis=None))
        error.append(error_i)
    return np.array(error)


def plot(elevation, angle, theta_1, theta_0, error, theta_1_vals, theta_0_vals, error_vals):
    # setting up the figure size
    fig = plt.figure(figsize=[12, 15])
    # setting up the axes as a 3 dimensional plot
    ax = fig.gca(projection='3d')
    # plot the surface
    ax.plot3D(theta_1_vals.flatten(), theta_0_vals.flatten(),
              error_vals.flatten(), 'orange', marker='.', linewidth=0.5)
    ax.plot_surface(theta_1, theta_0, error, cmap=cm.coolwarm)
    ax.set_xlabel('theta_1')
    ax.set_ylabel('theta_0')
    ax.set_zlabel('J(theta)')
    ax.view_init(elev=elevation, azim=angle)
    ax.set_title(
        'Surface Plot of Variation of J(theta) on Changing theta_1 and theta_0 Parameters')
    plt.savefig('./surface_plot.png')


def mesh_plot(theta_1_vals, theta_0_vals, error_vals, x, y):
    theta_1 = np.linspace(-1, 1, 100)
    theta_0 = np.linspace(0, 2, 100)
    theta_1, theta_0 = np.meshgrid(theta_1, theta_0)
    error = f(theta_1, theta_0, x, y)
    plot(65, 30, theta_1, theta_0, error,
         theta_1_vals, theta_0_vals, error_vals)
    #iplot = interactive(plot, elevation= (-90, 90, 5), angle = (-90, 90, 5))
    #plot(40, 75)

# PART (d) : PLOT


def contour_plot(theta_1_vals, theta_0_vals, title, x, y):
    fig = plt.figure(figsize=[12, 10])
    theta_1 = np.linspace(-1, 1, 100)
    theta_0 = np.linspace(0, 2, 100)
    theta_1, theta_0 = np.meshgrid(theta_1, theta_0)
    error = f(theta_1, theta_0, x, y)

    plt.title(title)
    plt.xlabel('theta_1')
    plt.ylabel('theta_0')

    plt.contour(theta_1, theta_0, error, 20, cmap='coolwarm')
    plt.plot(theta_1_vals, theta_0_vals,
             color='orange', marker='.', linewidth=0.5)
    plt.savefig('./'+title+'.png')
    plt.clf()

# PART (e) : PLOT


def contour_plots_eta(x, y):
    learning_rates = [0.001, 0.025, 0.1]
    delta = 10**(-10)

    for eta in learning_rates:
        final_theta, error_vals, theta_1_vals, theta_0_vals = linear_regression(
            x, y, eta, delta, False)
        theta_1_vals = np.array(theta_1_vals)
        theta_0_vals = np.array(theta_0_vals)
        contour_plot(theta_1_vals, theta_0_vals,
                     'Contour plot of parameters when learning rate = '+str(eta), x, y)


def main():
    x, y = load_data()
    #part (a)
    learning_rate = 0.01
    delta = 10**(-10)
    final_theta, error_vals, theta_1_vals, theta_0_vals = linear_regression(
        x, y, learning_rate, delta, False)
    print("Result of part (a): ")
    print("Learned parameters [theta_1, theta_0]: ", final_theta.T)
    # part(b)
    y_hat = np.dot(x.T, final_theta)
    scatter_plot(x, y, y_hat)
    # part(c)
    error_vals = np.array(error_vals)
    theta_1_vals = np.array(theta_1_vals)
    theta_0_vals = np.array(theta_0_vals)
    mesh_plot(theta_1_vals, theta_0_vals, error_vals, x, y)
    # part(d)
    contour_plot(theta_1_vals, theta_0_vals,
                 'Contour plot of parameters', x, y)
    # part(e)
    contour_plots_eta(x, y)


main()
