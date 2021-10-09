# ENTRY NUMBER: 2020CSY7575 (classes 5 and 6)

# Installing packages
# 1. conda install -c conda-forge cvxopt
# 2. pip install -U libsvm-official

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from libsvm.svmutil import *
import os
import time
import sys
argv = sys.argv[1:]

# READ DATA


def read_data(train_file, test_file, classes=None):
    # read train_data
    x_train = np.genfromtxt(train_file, delimiter=',')
    # read test_data
    x_test = np.genfromtxt(test_file, delimiter=',')

    # choose data from only given classes, here 5 and 6
    if(classes != None):
        # initialize list of indices, x_train indices where labels are classes[0] : (here 5)
        idx_train = list(np.argwhere(x_train[:, -1] == classes[0]).squeeze())
        idx_test = list(np.argwhere(x_test[:, -1] == classes[0]).squeeze())
        # append list of indices, x_train indices where labels are remaining classes to be selected : (here 6 only)
        for cls in classes[1:]:
            idx_train += list(np.argwhere(x_train[:, -1] == cls).squeeze())
            idx_test += list(np.argwhere(x_test[:, -1] == cls).squeeze())
        # shuffle indices to shuffle data of different classes randomly for validation examples, don't need to shuffle test data
        # split training data in the ration 80:20 for train:val data
        np.random.seed(0)
        np.random.shuffle(idx_train)
        # upper limit excluded
        x_val = x_train[idx_train[:int(0.2*len(idx_train))]]
        x_train = x_train[idx_train[int(0.2*len(idx_train)):]]
        x_test = x_test[idx_test]

    # prepare train_data
    m = x_train.shape[0]
    num_features = x_train.shape[1]-1
    # features
    images_train = x_train[:, 0:num_features]  # all rows, cols from 0 to 784
    images_train /= 255  # normalizing pixel values
    # labels
    labels_train = (x_train[:, num_features]).reshape(
        (m, 1))  # last column has label
    if(classes != None):
        for i in range(labels_train.shape[0]):
            if(labels_train[i][0] == classes[0]):
                labels_train[i][0] = 1
            else:
                labels_train[i][0] = -1

    # prepare val_data
    m = x_val.shape[0]
    num_features = x_val.shape[1]-1
    # features
    images_val = x_val[:, 0:num_features]  # all rows, cols from 0 to 784
    images_val /= 255  # normalizing pixel values
    # labels
    labels_val = (x_val[:, num_features]).reshape(
        (m, 1))  # last column has label
    if(classes != None):
        for i in range(labels_val.shape[0]):
            if(labels_val[i][0] == classes[0]):
                labels_val[i][0] = 1
            else:
                labels_val[i][0] = -1

    # prepare test_data
    m = x_test.shape[0]
    num_features = x_test.shape[1]-1
    # features
    images_test = x_test[:, 0:num_features]  # all rows, cols from 0 to 784
    images_test /= 255  # normalizing pixel values
    # labels
    labels_test = (x_test[:, num_features]).reshape(
        (m, 1))  # last column has label
    if(classes != None):
        for i in range(labels_test.shape[0]):
            if(labels_test[i][0] == classes[0]):
                labels_test[i][0] = 1
            else:
                labels_test[i][0] = -1

    # each example is should be a column, to get formulation as per class notes
    return images_train.T, labels_train.T, images_val.T, labels_val.T, images_test.T, labels_test.T


# SVM USING CVXOPT

def kernel_matrix(X, Z, kernel, gamma=0.05):
    if(kernel == 'linear'):
        return np.dot(X.T, Z)
    elif(kernel == 'gaussian'):
        # shape X and Z = #num_features*num_examples
        K = np.zeros((X.shape[1], Z.shape[1]))
        for i in range(X.shape[1]):
            for j in range(Z.shape[1]):
                K[i][j] = np.exp(-1*gamma *
                                 np.linalg.norm(X[:, i] - Z[:, j]) ** 2)

        return K


def train(X, Y, kernel, C):
    # shape X: num_features*num_examples, shape Y: 1*num_examples
    m = X.shape[1]
    P = matrix(np.multiply(np.dot(Y.T, Y), kernel_matrix(X, X, kernel)))
    q = matrix(-1*np.ones((m)).reshape((m, 1)))
    G = matrix(np.row_stack((-np.eye(m), np.eye(m))))
    h = matrix(np.row_stack(
        (np.zeros(m).reshape((m, 1)), C*np.ones(m).reshape((m, 1)))))
    A = matrix(Y)
    b = matrix(np.zeros(1).reshape((1, 1)))
    sol = solvers.qp(P, q, G, h, A, b)
    return sol


def get_params_support_vectors(sol, X, Y, C, kernel='linear'):
    alphas = sol['x']
    alphas = np.array(alphas)
    # print(alphas.shape) #num_eg*1
    # print(Y.shape) #1*num_eg
    threshold = 1e-5  # points lying on margin
    # computing support vectors
    S_idx = []
    for i in range(alphas.shape[0]):
        if alphas[i] > threshold and alphas[i] <= C:
            S_idx.append(i)
    S_idx = np.array(S_idx)
#     print(S_idx)
    if(kernel == 'linear'):
        # computing w
        w = np.dot(X, (alphas.T*Y).T)  # shape w: num_feature*1
        b = 1/len(S_idx) * np.sum(Y[:, S_idx] - np.dot(w.T, X[:, S_idx]))
    else:
        w = None
        K = kernel_matrix(X[:, S_idx], X[:, S_idx], 'gaussian')
        b = 1/len(S_idx) * np.sum(Y[:, S_idx] -
                                  np.dot(alphas[S_idx, :].T*Y[:, S_idx], K))
    print("b: ", b)
    return S_idx, alphas, w, b


def test(X, Y, w, b, X_train=None, Y_train=None, S_idx=None, alphas=None, kernel='linear'):
    # shape Y_hat: 1*num_instances
    if(kernel == 'linear'):
        Y_hat = np.dot(w.T, X) + b
    else:
        # shape: num_support_vectors*num_test_instances
        K = kernel_matrix(X_train[:, S_idx], X, 'gaussian')
        # 1*num_test_instances
        Y_hat = np.dot(alphas[S_idx, :].T*Y_train[:, S_idx], K) + b
    Y_pred = []
    for i in range(Y_hat.shape[1]):
        if(Y_hat[0][i] >= 0):
            Y_pred.append(1)
        else:
            Y_pred.append(-1)
    Y_pred = (np.array(Y_pred)).reshape((1, Y_hat.shape[1]))
    return Y_pred, accuracy(Y_pred, Y)


#  SVM USING LIBSVM

def train_libsvm(X, Y, kernel, C=1.0, gamma=0.05):
    # input format for libsvm: Y shape - (m,) , X shape: m*n
    prob = svm_problem(Y.reshape((Y.shape[1],)), X.T)
    if(kernel == 'linear'):
        param = svm_parameter('-s 0 -t 0 -c 1.0 -q')
    else:
        param = svm_parameter('-s 0 -t 2 -g 0.05 -c 1.0 -q')

    model = svm_train(prob, param)
    return model


def test_libsvm(X, Y, model):
    p_label, p_acc, p_val = svm_predict(
        Y.reshape((Y.shape[1],)), X.T, model, '-b 0 -q')
#     print("Labels: ", p_label)
#     print("Accuracy: ",p_acc)
    return p_label, p_acc[0]


def get_params_support_vectors_libsvm(model):
    sv_indices = model.get_sv_indices()
    nr_sv = model.get_nr_sv()
    support_vector_coefficients = model.get_sv_coef()
    print("b: ", support_vector_coefficients[int(nr_sv-1)])
#     print("Support vector indices: ", sv_indices)
#     print("Support vector count: ", nr_sv)
#     print("support_vector_coefficients: ", len(support_vector_coefficients))
    return sv_indices, nr_sv, support_vector_coefficients


#  UTILITY FUNCTIONS
def accuracy(Y_pred, Y):
    m = Y.shape[1]
    cnt = 0
    for i in range(Y.shape[1]):
        if(Y_pred[0][i] == Y[0][i]):
            cnt += 1
    return (cnt/m)*100


def plot_confusion_matrix(preds, y_test, img_name, mode='cvxopt'):
    if not os.path.exists('./Q2/Binary_Classification_Results'):
        os.makedirs('./Q2/Binary_Classification_Results')

    y_test_final = []
    y_preds = []
    # confusion matrix allows binary or multiclass setting only (so for -1 and 1 gives error)
    if(mode == 'libsvm'):
        m_pred = len(preds)
        preds = (np.array(preds)).reshape((1, m_pred))
    for i in range(preds.shape[1]):
        if(preds[0][i] == -1.0):
            y_preds.append(0)
        else:
            y_preds.append(1)

        if(y_test[0][i] == -1.0):
            y_test_final.append(0)
        else:
            y_test_final.append(1)

    conf_mat = confusion_matrix(y_test_final, y_preds)
    # print(conf_mat)
    df = pd.DataFrame(conf_mat,
                      index=[-1, 1],
                      columns=[-1, 1])
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('./Q2/Binary_Classification_Results/'+img_name+'.png')
    plt.show()


def compute_confusion_matrix(preds, y_test, mode='cvxopt'):
    y_test_final = []
    y_preds = []
    # confusion matrix allows binary or multiclass setting only (so for -1 and 1 gives error)
    if(mode == 'libsvm'):
        m_pred = len(preds)
        preds = (np.array(preds)).reshape((1, m_pred))

    for i in range(preds.shape[1]):
        if(preds[0][i] == -1.0):
            y_preds.append(0)
        else:
            y_preds.append(1)

        if(y_test[0][i] == -1.0):
            y_test_final.append(0)
        else:
            y_test_final.append(1)

    num_classes = len(np.unique(y_test))
    # print(num_classes)
    conf_mat = np.zeros((num_classes, num_classes))
    for i in range(len(y_test_final)):
        conf_mat[y_test_final[i]][y_preds[i]] += 1

    print_confusion_matrix(conf_mat)


def print_confusion_matrix(conf_mat):
    print("Self Implemented Confusion Matrix Result: \n")
    print(-1, 1, "<- predicted class")
    m = len(conf_mat)
    for i in range(m):
        for j in range(m):
            print(int(conf_mat[i][j]), end=' ')
        print()


def a(images_train, labels_train, images_val, labels_val, images_test, labels_test):
    print(
        "--------------------- PART A [CVXOPT with Linear Kernel] -----------------------\n")
    print("Training ...")
    start_time = time.time()
    sol = train(images_train, labels_train, 'linear', 1.0)
    print("Time taken to train(in sec): ", time.time() - start_time)
    S_idx, alphas, w, b = get_params_support_vectors(
        sol, images_train, labels_train, 1.0)

    print("Test and Validation ...")
    y_val_pred, val_acc = test(images_val, labels_val, w, b)
    y_pred, test_acc = test(images_test, labels_test, w, b)
    print("Number of support vectors at threshold = 1e-5: ", S_idx.shape[0])
    print("Validation accuracy(in %): ", val_acc)
    print("Test accuracy(in %): ", test_acc)
    compute_confusion_matrix(y_pred, labels_test)
    plot_confusion_matrix(y_pred, labels_test, 'confusion_matrix_a')


def b(images_train, labels_train, images_val, labels_val, images_test, labels_test):
    print(
        "\n--------------------- PART B [CVXOPT with Gaussian Kernel] -----------------------\n")
    print("Training ...")
    start_time = time.time()
    sol = train(images_train, labels_train, 'gaussian', 1.0)
    print("Time taken to train(in sec): ", time.time() - start_time)
    S_idx, alphas, w, b = get_params_support_vectors(
        sol, images_train, labels_train, 1.0, 'gaussian')

    print("Test and Validation ...")
    y_val_pred, val_acc = test(images_val, labels_val, w, b,
                               images_train, labels_train, S_idx, alphas, 'gaussian')
    y_pred, test_acc = test(images_test, labels_test, w, b,
                            images_train, labels_train, S_idx, alphas, 'gaussian')
    print("Number of support vectors at threshold = 1e-5: ", S_idx.shape[0])
    print("Validation accuracy: ", val_acc)
    print("Test accuracy: ", test_acc)
    compute_confusion_matrix(y_pred, labels_test)
    plot_confusion_matrix(y_pred, labels_test, 'confusion_matrix_b')


def c(images_train, labels_train, images_val, labels_val, images_test, labels_test):
    print(
        "\n--------------------- PART C(i) [LIBSVM with Linear Kernel] -----------------------\n")
    start_time = time.time()
    model = train_libsvm(images_train, labels_train, 'linear')
    print("Time taken to train(in sec): ", time.time() - start_time)
    y_val_pred, val_acc = test_libsvm(images_val, labels_val, model)
    y_pred, test_acc = test_libsvm(images_test, labels_test, model)
    sv_indices, nr_sv, support_vector_coefficients = get_params_support_vectors_libsvm(
        model)
    print("Number of support vectors: ", nr_sv)
    print("Validation accuracy (in %): ", val_acc)
    print("Test accuracy (in %): ", test_acc)
    compute_confusion_matrix(y_pred, labels_test, mode='libsvm')
    plot_confusion_matrix(y_pred, labels_test,
                          'confusion_matrix_c(i)', 'libsvm')

    # Part C(ii) - using LIBSVM (Gaussian Kernel)
    print(
        "\n--------------------- PART C(ii) [LIBSVM with Gaussian Kernel] -----------------------\n")
    start_time = time.time()
    model = train_libsvm(images_train, labels_train, 'gaussian')
    print("Time taken to train(in sec): ", time.time() - start_time)
    y_val_pred, val_acc = test_libsvm(images_val, labels_val, model)
    y_pred, test_acc = test_libsvm(images_test, labels_test, model)
    sv_indices, nr_sv, support_vector_coefficients = get_params_support_vectors_libsvm(
        model)
    print("Number of support vectors: ", nr_sv)
    print("Validation accuracy (in %): ", val_acc)
    print("Test accuracy (in %): ", test_acc)
    compute_confusion_matrix(y_pred, labels_test, mode='libsvm')
    plot_confusion_matrix(y_pred, labels_test,
                          'confusion_matrix_c(ii)', 'libsvm')

# Main FUNCTION


def main():
    # read the data
    print("Reading Data ...")
    train_file = argv[0]  # './mnist/train.csv'
    test_file = argv[1]  # './mnist/test.csv'
    images_train, labels_train, images_val, labels_val, images_test, labels_test = read_data(
        train_file, test_file, [5, 6])
    # print(images_train.shape, labels_train.shape, images_val.shape, labels_val.shape, images_test.shape, labels_test.shape)
    # Part A
    if(argv[2] == 'a'):
        a(images_train, labels_train, images_val,
          labels_val, images_test, labels_test)

    # Part B
    if(argv[2] == 'b'):
        b(images_train, labels_train, images_val,
          labels_val, images_test, labels_test)

    # Part C(i) - using LIBSVM (Linear Kernel)
    if(argv[2] == 'c'):
        c(images_train, labels_train, images_val,
          labels_val, images_test, labels_test)


main()


# REFERENCES
# 1. https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
# 2. http://cvxopt.org/applications/svm/
# 3. https://numpy.org/devdocs/reference/generated/numpy.row_stack.html
# 4. https://github.com/cjlin1/libsvm/blob/master/python/libsvm/svmutil.py
# 5. https://github.com/cjlin1/libsvm/tree/master/python
