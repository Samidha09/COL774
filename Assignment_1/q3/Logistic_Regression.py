''' REFERENCES
https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy '''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def load_data(feature_file, label_file):
    # read data from respective files
    X = np.genfromtxt(feature_file, delimiter=',')
    Y = np.genfromtxt(label_file, delimiter=',')

    # normalize x_2 and x_1 - assuming first column of input file as x_2, second as x_1
    x_1 = X[:, 1]
    x_2 = X[:, 0]
    x_1 = (x_1 - np.mean(x_1))/np.std(x_1)
    x_2 = (x_2 - np.mean(x_2))/np.std(x_2)

    # intercept term
    x_0 = np.ones(X.shape[0])

    # make final dataset by stacking features row wise : each column is an example
    final_X = np.row_stack((x_2, x_1))
    final_X = np.row_stack((final_X, x_0))

    Y = Y.reshape((Y.shape[0], 1))
#     print(final_X.shape, Y.shape)
    return final_X, Y


def plot(X, Y, theta):
    X = X.T
    sns.set_style("darkgrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 8))
    x_2 = X[:, 0]
    x_1 = X[:, 1]

    # for getting appropriate legend
    Y_labels = []
    for y in Y:
        Y_labels.append('Class 0' if y == 0 else 'Class 1')

    plt.title(
        'Scatter Plot Of Examples With Decision Boundary Learned By Logistic Regression')
    plt.xlabel('Feature 1 (x_1)')
    plt.ylabel('Feature 2 (x_2)')
    sns.scatterplot(x_1, x_2, hue=Y_labels)

    # theta: vector of shape (1*(num_features+1)), here num_features = 2
    y_hat = (-theta[0][2] - theta[0][1] * X[:, 1]) / theta[0][0]

    #sns.lineplot(x_1, y_hat)
    plt.plot(x_1, y_hat, color='#60b57e', linewidth=0.5)
    plt.savefig('plot.png')


def sigmoid(z):
    return 1/(1+np.exp(-z))


def one_minus_sigmoid(z):
    return (1 - sigmoid(z))


def cost(X, Y, theta):  # negative log likelihood (averaged)
    m = X.shape[1]
    phi = sigmoid(np.dot(X.T, theta))  # shape: num_eg*1
    val1 = np.log(phi)
    val2 = np.log(one_minus_sigmoid(phi))
    res1 = np.dot(Y.T, val1)  # shape: 1*1
    res2 = np.dot((1-Y).T, val2)  # shape: 1*1
    res = (-1)/m*(res1[0][0] + res2[0][0])
    return res


def gradient(theta, X, Y):
    m = X.shape[0]
    Y_hat = sigmoid(np.dot(X.T, theta))
    grad = 1/m * np.dot(X, (Y_hat - Y))  # for negative log likelihood
    return grad


def get_diagonal_matrix(theta, X):
    m = X.shape[1]
    diag = np.zeros((m, m))

    for i in range(m):
        V = X[:, i].reshape((X.shape[0], 1))
        phi = sigmoid(np.dot(V.T, theta))  # shape of phi: 1*1
        diag[i][i] = np.dot(phi.T, (1 - phi))

    return diag


def hessian(theta, X):
    D = get_diagonal_matrix(theta, X)
    res = X @ (D @ X.T)
    return res

# HELPER FUNCTIONS TO CHECK HESSIAN PROPERTIES


def is_semi_pos_def(H):
    return np.all(np.linalg.eigvals(H) >= 0)


def is_symmetric(H, rtol=1e-05, atol=1e-08):
    return np.allclose(H, H.T, rtol=rtol, atol=atol)


def check_H(H):  # return true if hessian is symmetric and positive semi-definite
    if(is_semi_pos_def(H) and is_symmetric(H)):
        return True
    return False


# shape of X: num_eg*num_features, shape of Y: num_eg*1
def logistic_regression(X, Y, epochs=1000, learning_rate=0.01, verbose=True, check_hessian=False):
    # vector of shape num_features*1, [[theta_2] ,[theta_1], [theta_0]]
    theta = np.zeros((X.shape[0], 1))
    timestep = 0
    while(timestep < epochs):
        cost_t = cost(X, Y, theta)
        H = hessian(theta, X)
        if(check_hessian):
            print(check_H(H))
        if(np.linalg.det(H) != 0):
            H_inv = np.linalg.inv(H)  # shape: num_features*num_features
            grad = gradient(theta, X, Y)  # shape: num_features
            theta = theta - (H_inv @ grad)
        else:  # take a step by gradient descent if hessian is not invertible
            theta = theta - learning_rate*gradient(theta, X, Y)

        cost_t_plus_1 = cost(X, Y, theta)
        timestep += 1

        if(abs(cost_t - cost_t_plus_1) < 10**(-8)):
            break
        if(verbose):
            print(timestep, " : ", cost(X, Y, theta))

    return theta


def main():
    # load data
    feature_file = '../Data/q3/logisticX.csv'
    label_file = '../Data/q3/logisticY.csv'
    X, Y = load_data(feature_file, label_file)

    # get theta (parameters)
    # make verbose=True to see cost after each update
    theta = logistic_regression(X, Y, verbose=False)
    final_theta = theta.reshape((1, theta.shape[0]))
    print("Final parameters: ", final_theta)
    print("Final Cost ", cost(X, Y, theta))
    # plot decision boundary
    plot(X, Y, final_theta)
    # plots are saved in the current directory and can be viewed from there


main()
