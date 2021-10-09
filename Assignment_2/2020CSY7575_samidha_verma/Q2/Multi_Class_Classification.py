import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix
from itertools import combinations
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from libsvm.svmutil import *
import os
import time
import sys
argv = sys.argv[1:]
# print(argv)
# READ DATA


def read_data(train_file, test_file, classes=None):
    # read train_data
    x_train = np.genfromtxt(train_file, delimiter=',')
    # read test_data
    x_test = np.genfromtxt(test_file, delimiter=',')

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
    else:
        np.random.seed(0)
        np.random.shuffle(x_train)
        x_val = x_train[0:int(0.2*x_train.shape[0]), :]  # upper limit excluded
        x_train = x_train[int(0.2*x_train.shape[0]):, :]

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
    return S_idx, alphas, w, b


def test(X, Y, w, b, classes=None, X_train=None, Y_train=None, S_idx=None, alphas=None, kernel='linear'):
    # shape Y_hat: 1*num_instances
    if(kernel == 'linear'):
        Y_hat = np.dot(w.T, X) + b
    else:
        # shape: num_support_vectors*num_test_instances
        K = kernel_matrix(X_train[:, S_idx], X, 'gaussian')
        # 1*num_test_instances
        Y_hat = np.dot(alphas[S_idx, :].T*Y_train[:, S_idx], K) + b

    Y_pred = []
    Y_pred_score = []

    for i in range(Y_hat.shape[1]):
        if(Y_hat[0][i] >= 0):
            Y_pred.append(classes[0])
        else:
            Y_pred.append(classes[1])

        Y_pred_score.append(Y_hat[0][i])

    Y_pred = (np.array(Y_pred)).reshape((1, Y_hat.shape[1]))
    Y_pred_score = (np.array(Y_pred_score)).reshape((1, Y_hat.shape[1]))

    return Y_pred, Y_pred_score


def train_multiclass(X, Y, train_file, test_file, saved, C=1.0, data_dict_saved=False):

    if(data_dict_saved == False):
        print("Pre-processing data for training ....")
        start = time.time()
        classes = np.unique(Y)
        pairs_classes = list(combinations(classes, 2))
        data_dict = {}
        for pair in pairs_classes:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = read_data(
                train_file, test_file, classes=pair)
            data_dict[pair] = (X_train, Y_train)

        print("Time taken for data pre-processing: ", time.time() - start)

        f = open('./Q2/data_dict.pickle', 'wb')
        pickle.dump(data_dict, f)
        f.close()
    else:
        f = open('./Q2/data_dict.pickle', 'rb')
        data_dict = pickle.load(f)
        f.close()

    if(saved == True):
        nf = open('./Q2/sv_alpha_param_dict.pickle', 'rb')
        res_dict = pickle.load(nf)
        nf.close()
    else:
        res_dict = {}
        for pair in data_dict.keys():
            data = data_dict[pair]
            X_train, Y_train = data[0], data[1]
            sol = train(X_train, Y_train, 'gaussian', 1.0)
            S_idx, alphas, w, b = get_params_support_vectors(
                sol, X_train, Y_train, 1.0, 'gaussian')
            res_dict[pair] = (S_idx, alphas, w, b)
            print("Done: ", pair)

        nf = open('./Q2/sv_alpha_param_dict.pickle', 'wb')
        pickle.dump(res_dict, nf)
        nf.close()

    return res_dict, data_dict


def test_multiclass(X_test, Y_test, X_val, Y_val, model, data_dict, saved):
    if(saved == True):
        f1 = open('./Q2/results_multiclass.pickle', 'rb')
        result = pickle.load(f1)
        f1.close()

        return result

    Y_pred = []
    num_classes = len(np.unique(Y_test))
    test_m = X_test.shape[1]
    val_m = X_val.shape[1]
    votes_val = np.zeros((val_m, num_classes))
    votes_val_score = np.zeros((val_m, num_classes))
    votes_test = np.zeros((test_m, num_classes))
    votes_test_score = np.zeros((test_m, num_classes))

    for pair in model.keys():
        params = model[pair]
        data = data_dict[pair]
        X_train, Y_train = data[0], data[1]
        S_idx, alphas, w, b = params[0], params[1], params[2], params[3]
        val_pred, val_pred_score = test(
            X_val, Y_val, w, b, pair, X_train, Y_train, S_idx, alphas, 'gaussian')
        test_pred, test_pred_score = test(
            X_test, Y_test, w, b, pair, X_train, Y_train, S_idx, alphas, 'gaussian')
        # print(val_pred.shape, val_pred_score.shape) #1*4000
        for i in range(val_pred.shape[1]):
            prediction = val_pred[:, i].squeeze()
            # print(prediction)
            votes_val[i][int(prediction)] += 1
            votes_val_score[i][int(prediction)] += abs(val_pred_score[:, i])

        for i in range(test_pred.shape[1]):
            prediction = test_pred[:, i]
            votes_test[i][int(prediction)] += 1
            votes_test_score[i][int(prediction)] += abs(test_pred_score[:, i])

    result = (votes_val, votes_val_score, votes_test, votes_test_score)

    # saving results
    f1 = open('./Q2/results_multiclass.pickle', 'wb')
    pickle.dump(result, f1)
    f1.close()

    return result


def final_predictions_multiclass(results):
    votes_val, votes_val_score, votes_test, votes_test_score = results[
        0], results[1], results[2], results[3]
    val_pred_final = []
    test_pred_final = []

    for i in range(votes_val.shape[0]):
        prediction = np.argmax(votes_val[i, :])
        max_idx = list(np.argwhere(
            votes_val[i, :] == votes_val[i][prediction]))
        if(len(max_idx) > 1):  # breaking ties with score
            prediction = np.argmax(votes_val_score[i, :])
        val_pred_final.append(prediction)

    for i in range(votes_test.shape[0]):
        prediction = np.argmax(votes_test[i, :])
        max_idx = list(np.argwhere(
            votes_test[i, :] == votes_test[i][prediction]))
        if(len(max_idx) > 1):  # breaking ties with score
            prediction = np.argmax(votes_test_score[i, :])
        test_pred_final.append(prediction)

    val_pred_final = np.array(val_pred_final).reshape((1, len(val_pred_final)))
    test_pred_final = np.array(test_pred_final).reshape(
        (1, len(test_pred_final)))

    return val_pred_final, test_pred_final


# SVM USING LIBSVM
def train_libsvm(X, Y, saved=False):
    if(saved == True):
        model = svm_load_model('./Q2/libsvm_part_2b.model')
        return model
    # input format for libsvm: Y shape - (m,) , X shape: m*n
    prob = svm_problem(Y.reshape((Y.shape[1],)), X.T)
    param = svm_parameter('-s 0 -t 2 -q -g 0.05')  # C=1 (default)

    model = svm_train(prob, param)
    svm_save_model('./Q2/libsvm_part_2b.model', model)
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
#     print("Support vector indices: ", sv_indices)
#     print("Support vector count: ", nr_sv)
    # print("support_vector_coefficients: ", len(support_vector_coefficients))
    return sv_indices, nr_sv, support_vector_coefficients


# K-fold cross validation - LIBSVM
def train_libsvm_kfold(X, Y, X_test, Y_test, K):
    prob_train = svm_problem(Y.reshape((Y.shape[1],)), X.T)

    C = [1e-5, 1e-3, 1, 5, 10]
    cross_val_acc = []
    test_acc = []
    for c in C:
        # performing cross validation
        param = svm_parameter(
            '-s 0 -t 2 -q -g 0.05 -c ' + str(c) + ' -v '+str(K))
        CV_ACC = svm_train(prob_train, param)
        cross_val_acc.append(CV_ACC)
        # re-training for test acc, since cross validation mode does not return the model in libsvm, finally choose model with best cv acc
        param = svm_parameter('-s 0 -t 2 -q -g 0.05 -c ' + str(c))
        model = svm_train(prob_train, param)
        p_label, p_acc, p_val = svm_predict(Y_test.reshape(
            (Y_test.shape[1],)), X_test.T, model, '-b 0 -q')
        test_acc.append(p_acc[0])

    return C, cross_val_acc, test_acc


# UTILITY FUNCTION
def accuracy(Y_pred, Y):
    m = Y.shape[1]
    cnt = 0
    for i in range(Y.shape[1]):
        if(Y_pred[0][i] == Y[0][i]):
            cnt += 1
    return (cnt/m)*100


def plot_confusion_matrix(preds, y_test, mode='cvxopt'):
    if not os.path.exists('./Q2/Multiclass_Classification_Results'):
        os.makedirs('./Q2/Multiclass_Classification_Results')
    #y_test = list(y_test)
    # confusion_matrix(y_test, preds)
    conf_mat = compute_confusion_matrix(preds, y_test, mode)
    df = pd.DataFrame(conf_mat,
                      index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".1f")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(
        './Q2/Multiclass_Classification_Results/multi_class_confusion_matrix_'+mode+'.png')
    plt.show()

    return conf_mat


def plot_cv_res(C, cross_val_acc, test_acc):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.title('Cross validation and Test accuracy vs log(C) value')
    plt.xlabel('log(C)')
    plt.ylabel('accuracy (in %)')
    log_C = list(np.log(np.array(C)))
    plt.plot(log_C, cross_val_acc, linewidth=1.0,
             label='Cross-validation accuracy')
    plt.plot(log_C, test_acc, linewidth=1.0, label='Test accuracy')
    plt.legend()
    plt.savefig('./Q2/cv_test_acc_vs_logC.png')
    plt.show()


def compute_confusion_matrix(preds, y_test, mode='cvxopt'):
    y_test_final = []
    y_preds = []

    if(mode == 'libsvm'):
        m_pred = len(preds)
        preds = (np.array(preds)).reshape((1, m_pred))

    for i in range(preds.shape[1]):
        y_preds.append(int(preds[:, i][0]))
        y_test_final.append(int(y_test[:, i][0]))

    num_classes = len(np.unique(y_test))
    # print(num_classes)
    conf_mat = np.zeros((num_classes, num_classes))
    for i in range(len(y_test_final)):
        conf_mat[y_test_final[i]][y_preds[i]] += 1

    # print_confusion_matrix(conf_mat)

    return conf_mat


# def print_confusion_matrix(conf_mat):
#     print("0, 1, 2, 3, 4, 5, 6, 7, 8, 9 <- predicted class")
#     m = len(conf_mat)
#     for i in range(m):
#         print("[", end=' ')
#         for j in range(m-1):
#             print(int(conf_mat[i][j]), end=', ')
#         print(int(conf_mat[i][m-1]), "]")


def print_misclassified_image_examples(X, Y, Y_pred, mode='cvxopt'):
    if not os.path.exists('./Q2/Misclassified_Test_Images_'+mode):
        os.makedirs('./Q2/Misclassified_Test_Images_'+mode)

    if(mode == 'libsvm'):
        m_pred = len(Y_pred)
        Y_pred = (np.array(Y_pred)).reshape((1, m_pred))

    misclassified_idx = []
    for i in range(Y.shape[1]):
        if(int(Y_pred[:, i][0]) != int(Y[:, i][0])):
            misclassified_idx.append(i)

    # plot 10 samples of misclassified images in test set
    np.random.seed(1996)
    img_samples = np.random.choice(misclassified_idx, 10)

    for i in img_samples:
        print("Actual Class: ", int(Y[:, i][0]))
        print("Predicted Class: ", int(Y_pred[:, i][0]))
        img = X[:, i].reshape((28, 28))
        plt.imshow(img, aspect='auto')
        label_img = 'Act_label' + \
            str(int(Y[:, i][0]))+'_Pred_label'+str(int(Y_pred[:, i][0]))+'.png'
        plt.savefig('./Q2/Misclassified_Test_Images_'+mode+'/'+label_img)
        # plt.show()
    print("Please check results in " +
          './Q2/Misclassified_Test_Images_'+mode+' folder.')


def f1_score(conf_mat, average=None):
    # taken 1 extra to keep 0th row and column empty
    num_classes = conf_mat.shape[0]
    prec_recall_dict = {}
    f1_dict = {}

    for i in range(num_classes):
        row_i = conf_mat[i, :]
        col_i = conf_mat[:, i]
        precision = conf_mat[i][i]/np.sum(row_i)
        recall = conf_mat[i][i]/np.sum(col_i)
        prec_recall_dict[i] = (precision, recall)
        # print(i, ":", prec_recall_dict[i])
        f1_dict[i] = (2*precision*recall)/(precision + recall)

    if(average == 'macro'):
        macro_f1 = 0
        for key in f1_dict.keys():
            macro_f1 += f1_dict[key]
        macro_f1 /= len(f1_dict.keys())
        print("Macro averaged F1-Score: ", macro_f1)
    else:
        for key in f1_dict.keys():
            print("Class: ", key, " -> F1-Score: ", f1_dict[key])


def a(images_train, labels_train, images_val, labels_val, images_test, labels_test, train_file, test_file, saved, data_saved=False):
    print(
        "--------------------- PART A [CVXOPT with Linear Kernel] -----------------------\n")
    # since training and testing takes a lot of time, loading stored parameters and results for testing code quickly
    print("Training....\n")
    start_time = time.time()
    model, data_dict = train_multiclass(
        images_train, labels_train, train_file, test_file, saved=saved, data_dict_saved=data_saved)
    if(saved == False):
        print("Time taken to train(in sec): ", time.time() - start_time)
    print("Test and Validation .....\n")
    start_time = time.time()
    results = test_multiclass(images_test, labels_test,
                              images_val, labels_val, model, data_dict, saved=saved)
    if(saved == False):
        print("Time taken to validate and test(in sec): ",
              time.time() - start_time)

    val_pred_final, test_pred_final = final_predictions_multiclass(results)
    val_acc = accuracy(val_pred_final, labels_val)
    test_acc = accuracy(test_pred_final, labels_test)
    print("Validation accuracy(in %): ", val_acc)
    print("Test accuracy(in %): ", test_acc)
    return test_pred_final


def b(images_train, labels_train, images_val, labels_val,
      images_test, labels_test, saved=False):
    print(
        "\n--------------------- PART (ii) [LIBSVM with Gaussian Kernel] -----------------------\n")
    print("Training....\n")
    start_time = time.time()
    model = train_libsvm(images_train, labels_train, saved=saved)
    if(saved == False):
        print("Time taken to train(in sec): ", time.time() - start_time)
    print("Test and Validation .....\n")
    y_pred_val, val_acc = test_libsvm(images_val, labels_val, model)
    y_pred, test_acc = test_libsvm(images_test, labels_test, model)
    sv_indices, nr_sv, support_vector_coefficients = get_params_support_vectors_libsvm(
        model)
    print("Number of support vectors: ", nr_sv)
    print("Validation accuracy (in %): ", val_acc)
    print("Test accuracy (in %): ", test_acc)

    return y_pred


def c(images_train, labels_train, images_val, labels_val,
      images_test, labels_test, train_file, test_file, saved):
    print("\n--------------------- PART (iii) -----------------------\n")
    # CVXOPT
    print("Getting test results from CVXOPT....")
    test_pred_final = a(images_train, labels_train, images_val, labels_val,
                        images_test, labels_test, train_file, test_file, saved=saved, data_saved=True)
    print("Confusion Matrix (results of test data using CVXOPT based SVM classifier)\n")
    #conf_mat = compute_confusion_matrix(test_pred_final, labels_test, 'cvxopt')
    conf_mat = plot_confusion_matrix(test_pred_final, labels_test, 'cvxopt')
    print("F1-Scores (results of test data using CVXOPT based SVM classifier)\n")
    f1_score(conf_mat)
    f1_score(conf_mat, 'macro')
    print("\nMisclassified Image Examples (from test data using CVXOPT based SVM classifier)\n")
    print_misclassified_image_examples(
        images_test, labels_test, test_pred_final, 'cvxopt')
    # LIBSVM
    print("Getting test results from LIBSVM....")
    y_pred = b(images_train, labels_train, images_val, labels_val,
               images_test, labels_test, saved=True)
    print("Confusion Matrix (results of test data using LIBSVM based SVM classifier)\n")
    #conf_mat = compute_confusion_matrix(y_pred, labels_test, 'libsvm')
    conf_mat = plot_confusion_matrix(y_pred, labels_test, 'libsvm')
    print("F1-Scores (results of test data using LIBSVM based SVM classifier)\n")
    f1_score(conf_mat)
    f1_score(conf_mat, 'macro')
    print("\nMisclassified Image Examples (from test data using LIBSVM based SVM classifier)\n")
    print_misclassified_image_examples(
        images_test, labels_test, y_pred, 'libsvm')


def d(images_train, labels_train, images_test, labels_test):
    print("\n--------------------- PART (iv) -----------------------\n")
    start_time = time.time()
    C, cross_val_acc, test_acc = train_libsvm_kfold(
        images_train, labels_train, images_test, labels_test, 5)
    print("Time taken for k-fold cross validation and testing (in sec): ",
          time.time() - start_time)
    # print(C, cross_val_acc, test_acc)
    plot_cv_res(C, cross_val_acc, test_acc)

# MAIN FUNCTION


def main():
    print("Reading Data .... ")
    train_file = argv[0]  # './mnist/train.csv'
    test_file = argv[1]  # './mnist/test.csv'
    images_train, labels_train, images_val, labels_val, images_test, labels_test = read_data(
        train_file, test_file)
    # print(images_train.shape, labels_train.shape, images_val.shape, labels_val.shape, images_test.shape, labels_test.shape)
    # PART a
    if(argv[2] == 'a'):
        test_pred_final = a(images_train, labels_train, images_val, labels_val,
                            images_test, labels_test, train_file, test_file, True)
    if(argv[2] == 'b'):
        y_pred = b(images_train, labels_train, images_val, labels_val,
                   images_test, labels_test)
    if(argv[2] == 'c'):
        c(images_train, labels_train, images_val, labels_val,
          images_test, labels_test, train_file, test_file, True)
    if(argv[2] == 'd'):
        d(images_train, labels_train, images_test, labels_test)


main()
