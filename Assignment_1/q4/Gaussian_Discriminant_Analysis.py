import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def load_data(feature_file, label_file):
    # read data from respective files
    X = np.genfromtxt(feature_file, dtype=(int, int))
    Y = np.genfromtxt(label_file, dtype=str)

    # normalize x_2 and x_1
    x_1 = X[:, 0]
    x_2 = X[:, 1]
    x_1 = (x_1 - np.mean(x_1))/np.std(x_1)
    x_2 = (x_2 - np.mean(x_2))/np.std(x_2)

    # make final dataset by stacking features column wise
    final_X = np.column_stack((x_1, x_2))
    Y = Y.reshape((Y.shape[0], 1))

    #print(final_X.shape, Y.shape)
    return final_X, Y

# function to convert class Alaska - 0 and Canada - 1


def to_num_class(Y_label):
    label_dict = {"Alaska": 0, "Canada": 1}
    Y = []
    for i in range(Y_label.shape[0]):
        Y.append(label_dict[Y_label[i][0]])
    Y = np.array(Y).reshape((Y_label.shape[0], 1))
    return Y


def convert_data_to_dataframe(X, Y_label):
    df = pd.DataFrame(X, columns=['Feature 2 (x_2)', 'Feature 1 (x_1)'])
    df['Y_label'] = Y_label
    return df


def get_linear_decision_boundary(X, Y):
    # equation of line: mX+c = 0
    phi, mu0, mu1, sigma = gda(X, Y, 'linear')
    # to avoid unprecedented broadcasting
    mu0 = mu0.reshape((1, mu0.shape[0]))
    mu1 = mu1.reshape((1, mu1.shape[0]))

    sigma_inv = np.linalg.inv(sigma)

    c = 1/2 * ((mu1 @ sigma_inv @ mu1.T)) - \
        (mu0 @ sigma_inv @ mu0.T) + np.log((1-phi)/phi)
    m = (mu0 - mu1) @ sigma_inv @ X.T
    mx = np.dot(m, X)
    y = -(mx[0][1]*X[:, 1] + c)/mx[0][0]
    return X[:, 1], y.reshape(y.shape[1],)


def get_quadratic_decision_boundary(X, Y):
    # equation of line: aX^2 + bX +c = 0
    phi, mu0, mu1, sigma0, sigma1 = gda(X, Y, 'quadratic')
    # to avoid unprecedented broadcasting
    mu0 = mu0.reshape((1, mu0.shape[0]))
    mu1 = mu1.reshape((1, mu1.shape[0]))

    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    sigma0_det = np.linalg.det(sigma0)
    sigma1_det = np.linalg.det(sigma1)
    # comput c
    C = ((1-phi)/phi) * math.sqrt((sigma1_det/sigma0_det))
    c = 1/2 * ((mu1 @ sigma1_inv @ mu1.T)) - \
        (mu0 @ sigma0_inv @ mu0.T) + np.log(C)
    # compute b
    b = (mu0 @ sigma0_inv - mu1 @ sigma1_inv)
    # compute a
    a = 1/2 * (sigma1_inv - sigma0_inv)
    return a, b, c


def boundary(x, a, b, c):
    return x.T @ a @ x + b @ x + c


def plot_training_data(X, Y_label, Y, decision_boundary='none'):
    sns.set_style("darkgrid")
    # sns.set_palette("Set2")
    # sns.set(rc={'figure.figsize':(14,10)})

    df = convert_data_to_dataframe(X, Y_label)
    sns.lmplot(x='Feature 1 (x_1)', y='Feature 2 (x_2)', data=df, fit_reg=False, hue='Y_label', legend=False, markers=[
               "o", "^"], palette=dict(Alaska="#63b5d6", Canada="#054215"), height=8, aspect=1.5)
    plt.legend(loc='upper right')
    plt.xlabel('Feature 1 (x_1)')
    plt.ylabel('Feature 2 (x_2)')

    if(decision_boundary == 'linear'):
        plot_x, plot_y = get_linear_decision_boundary(X, Y)
        plt.plot(plot_x, plot_y, color='#f0ea4f', linewidth=0.5)
        plt.title('Linear Decision boundary')
        plt.savefig('linear_boundary.png')
    elif(decision_boundary == 'quadratic'):
        plot_x, plot_y = get_linear_decision_boundary(X, Y)
        plt.plot(plot_x, plot_y, color='#f0ea4f', linewidth=0.5)
        # making quadratic decision boundary
        a, b, c = get_quadratic_decision_boundary(X, Y)
        x1, x2 = np.mgrid[-2.8:3:100j, -2:3.5:100j]
        x_s = np.c_[x1.flatten(), x2.flatten()]
        quadratic_boundary = np.array(
            [boundary(x, a, b, c) for x in x_s]).reshape(x1.shape)
        plt.contour(x1, x2, quadratic_boundary, [0])
        plt.title('Quadratic Decision boundary')
        plt.savefig('quadratic_boundary.png')
    else:
        plt.title('Scatter Plot of Training Examples')
        plt.savefig('scatter_plot.png')
    return


def phi(Y):
    m = Y.shape[0]
    return ((Y == 1).sum())/m


def mu_0(X, Y):
    m = (Y == 0).sum()
    relevant_Xs = X*(Y == 0)
    mean_arr = (np.sum(relevant_Xs, axis=0))/float(m)
    return mean_arr


def mu_1(X, Y):
    m = (Y == 1).sum()
    relevant_Xs = X*(Y == 1)
    mean_arr = (np.sum(relevant_Xs, axis=0))/float(m)
    return mean_arr


def sigma_0(X, Y):
    # outer product is col vec*row vec, since here each x_i is a row vector therefore we will do (x_i - MU_0)^T(x_i - MU_0)
    MU_0 = mu_0(X, Y)
    m = (Y == 0).sum()
    dim = X.shape[1]
    zero_idx = np.where(Y == 0)
    var_arr = np.zeros((dim, dim))

    for idx in zero_idx[0]:
        vec = X[idx] - MU_0
        outer_product = np.outer(vec, vec)
        var_arr += outer_product

    var_arr /= m
    return var_arr


def sigma_1(X, Y):
    MU_1 = mu_1(X, Y)
    m = (Y == 1).sum()
    dim = X.shape[1]
    one_idx = np.where(Y == 1)
    var_arr = np.zeros((dim, dim))

    for idx in one_idx[0]:
        vec = X[idx] - MU_1
        outer_product = np.outer(vec, vec)
        var_arr += outer_product

    var_arr /= m
    return var_arr


def sigma(X, Y):
    MU_0 = mu_0(X, Y)
    MU_1 = mu_1(X, Y)
    m = Y.shape[0]
    dim = X.shape[1]
    var_arr = np.zeros((dim, dim))

    for idx in range(m):
        if(Y[idx] == 0):
            vec = X[idx] - MU_0
        else:
            vec = X[idx] - MU_1
        vec = vec.reshape((dim, 1))
        outer_product = np.outer(vec, vec)
        var_arr += outer_product

    var_arr /= m
    return var_arr


def gda(X, Y, case='quadratic'):
    if(case == 'linear'):
        return phi(Y), mu_0(X, Y), mu_1(X, Y), sigma(X, Y)
    return phi(Y), mu_0(X, Y), mu_1(X, Y), sigma_0(X, Y), sigma_1(X, Y)


def main():
    # load data
    feature_file = '../Data/q4/q4x.dat'
    label_file = '../Data/q4/q4y.dat'
    X, Y_label = load_data(feature_file, label_file)
    Y = to_num_class(Y_label)
    # Call GDA for linear case:
    phi, mu0, mu1, sigma = gda(X, Y, 'linear')
    print("Results of part (a): ")
    print("phi", phi)
    print("mean -> mu0, mu1: ", mu0, mu1)
    print("cov:", sigma)
    print("----------------------------\n")
    # plot training data - part(b)
    plot_training_data(X, Y_label, Y)
    # plot linear decision boundary - part(c)
    plot_training_data(X, Y_label, Y, 'linear')
    # Call for GDA in general case
    phi, mu0, mu1, sigma0, sigma1 = gda(X, Y, 'quadratic')
    print("Results of part (d): ")
    print("phi", phi)
    print("mean -> mu0, mu1: ", mu0, mu1)
    print("cov0:", sigma0)
    print("cov1:", sigma1)
    print("----------------------------\n")
    # plot quadratic decision boundary - part(e)
    plot_training_data(X, Y_label, Y, 'quadratic')


main()
