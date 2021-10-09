import json
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams
import sys

argv = sys.argv[1:]
# print(argv)

#  LOAD AND PRE-PROCESS DATA


def read_data(train_file_name, test_file_name):
    print("Reading data ...")
    train_data = pd.read_json(train_file_name, lines=True)
    test_data = pd.read_json(test_file_name, lines=True)
    # final data
    # TRAIN DATA
    x_train = train_data['reviewText']
    y_train = train_data['overall']
    # TEST DATA
    x_test = test_data['reviewText']
    y_test = test_data['overall']

    return x_train, y_train, x_test, y_test


def summary_data(train_file_name, test_file_name):
    print("Reading data ...")
    train_data = pd.read_json(train_file_name, lines=True)
    test_data = pd.read_json(test_file_name, lines=True)
    # final data
    # TRAIN DATA
    x_train = train_data['summary']
    y_train = train_data['overall']
    # TEST DATA
    x_test = test_data['summary']
    y_test = test_data['overall']

    return x_train, y_train, x_test, y_test


# CLEAN DATA
def remove_punctution(tokens):
    words = []
    for word in tokens:
        if word.isalpha():
            # to not treat uppercase words differently
            words.append(word.lower())
    return words


def clean_data(tokens):
    words = []
    stop_words = stopwords.words('english')
    # print(stop_words)
    lemmatizer = WordNetLemmatizer()
    # porter = PorterStemmer()
    for word in tokens:
        if word not in stop_words:  # removing stop words
            words.append(lemmatizer.lemmatize(word))
            # words.append(porter.stem(word))

    # Stemming refers to the process of reducing each word to its root or base.
    # I will be doing lemmatization rather than stemming, because lemmatization of words is based on linguistics and words are more meaningful.
    return words

# FEATURE ENGINEERING : N-GRAMS


def ngrams(tokens, N):  # N = list of which n-grams to take
    ngram_features = [tokens[i:i+N[0]] for i in range(len(tokens)-N[0]+1)]
    for j in range(1, len(N)):
        for i in range(len(tokens)-N[j]+1):
            token = tokens[i:i+N[j]]
            ngram_features.append(token)
    for i in range(len(ngram_features)):
        ngram_features[i] = ' '.join(ngram_features[i])

    return ngram_features


# PREPARE REQUIRED DICTIONARIES
def dictionary_prepare(x_train, y_train, saved=True, clean=False, NGRAM=False, N=[2], only_ngram=False, part='1_a'):
    if(saved == True):
        # load dict from pickle file
        f1 = open('./Q1/class_words_dict_'+part+'.pickle', 'rb')
        class_words_dict = pickle.load(f1)
        f1.close()

        # load vocab from pickle file
        f2 = open('./Q1/vocabulary_'+part+'.pickle', 'rb')
        vocab = pickle.load(f2)
        f2.close()

        # load vocab from pickle file
        f3 = open('./Q1/class_num_words_dict_'+part+'.pickle', 'rb')
        class_num_words_dict = pickle.load(f3)
        f3.close()

        return vocab, class_words_dict, class_num_words_dict

    # set of all disctinct words in the training data
    vocab = set()
    # number of examples in training data
    m = len(y_train)
    # making dictionary of words per class: key=class, val=dict(key=word, val=frequency)
    class_words_dict = {}
    # total words in class key: key=class, val=sum of total number of words in all examples of class key
    class_num_words_dict = {}

    for i in range(m):
        doc = x_train[i]
        cls = y_train[i]  # class

        # split doc into list of individual words such that punctuations are kept separate from word
        tokens = word_tokenize(doc)
        # removing punctuations to get final list of words
        tokens = remove_punctution(tokens)
        # further do stemming, removing stopwords etc. for part (d)
        if(clean):
            tokens = clean_data(tokens)
        # if ngram model is to be used
        if(NGRAM):
            Ngram = ngrams(tokens, N)
            if(only_ngram == True):
                tokens = []
            for ngram in Ngram:
                tokens.append(ngram)

        # calculating total number of words of class cls
        if(cls in class_num_words_dict.keys()):
            class_num_words_dict[cls] += len(tokens)
        else:
            class_num_words_dict[cls] = len(tokens)

        for word in tokens:
            vocab.add(word)
            if(cls in class_words_dict.keys()):
                if(word in class_words_dict[cls].keys()):
                    # if word is present, increase frequency by 1
                    class_words_dict[cls][word] += 1
                else:  # make frequency of word 1 since word encountered for the first time for class 'cls'
                    class_words_dict[cls][word] = 1
            else:  # class is encountered for the first time
                # initialize dictionary at class 'cls' as key
                class_words_dict[cls] = {}
                # since dictionary is newly initialized, word can't possibly exist in it, therefore no need to check
                # set frequency to 1
                class_words_dict[cls][word] = 1
    # save dict to pickle file
    fp = open('./Q1/class_words_dict_'+part+'.pickle', 'wb')
    pickle.dump(class_words_dict, fp)
    fp.close()

    # save vocab to pickle file
    f = open('./Q1/vocabulary_'+part+'.pickle', 'wb')
    pickle.dump(vocab, f)
    f.close()

    # save class_num_words_dict to pickle file
    nf = open('./Q1/class_num_words_dict_'+part+'.pickle', 'wb')
    pickle.dump(class_num_words_dict, nf)
    nf.close()

    return vocab, class_words_dict, class_num_words_dict


# TRAIN
def train(x_train, y_train, vocab, class_words_dict, class_num_words_dict, alpha=1.0, saved=True, part='1_a'):
    if(saved == True):  # if parameters have already been saved, load them
        # load dict from pickle file
        f4 = open('./Q1/phi_' + part + '.pickle', 'rb')
        phi = pickle.load(f4)
        f4.close()

        # load vocab from pickle file
        f5 = open('./Q1/word_probs_per_class_' + part + '.pickle', 'rb')
        word_probs_per_class = pickle.load(f5)
        f5.close()

        return phi, word_probs_per_class

    # number of examples in training data
    m = len(y_train)

    # parameters
    phi = {}  # key:class, val:prob of class = num of examples with class key/total number of examples
    # key:class, val:dict(key=word, val=prob of word occuring in class key)
    word_probs_per_class = {}

    # calculating phi params
    for i in range(m):
        if(y_train[i] in phi.keys()):
            phi[y_train[i]] += 1  # increase frequency of class
        else:
            phi[y_train[i]] = 1
    # change frequency to probability
    for key in phi.keys():
        phi[key] /= m
        phi[key] = np.log(phi[key])  # taking log to prevent underflow

    # calculating word_probs_per_class
    mod_v = len(vocab)

    for cls in class_num_words_dict.keys():
        total_words = class_num_words_dict[cls]
        words_freqs = class_words_dict[cls]
        word_probs_per_class[cls] = {}
        for word in vocab:  # also use laplace smoothing with alpha hyperparameter
            numerator = alpha
            if(word in words_freqs.keys()):  # word occured in class cls
                numerator += words_freqs[word]
            denominator = (mod_v*alpha) + total_words
            prob_word = numerator/denominator
            # update probability in word_probs_per_class
            word_probs_per_class[cls][word] = np.log(
                prob_word)  # taking log to prevent underflow

    # save parameters
    f6 = open('./Q1/phi_' + part + '.pickle', 'wb')
    pickle.dump(phi, f6)
    f6.close()

    # save vocab to pickle file
    f7 = open('./Q1/word_probs_per_class_' + part + '.pickle', 'wb')
    pickle.dump(word_probs_per_class, f7)
    f7.close()

    return phi, word_probs_per_class


# TEST
def predict(x_test, y_test, phi, word_probs_per_class, mod_v, clean=False, NGRAM=False, N=[2], only_ngram=False):
    m = len(y_test)
    preds = []
    num_classes = len(phi.keys())
    classes = list(phi.keys())
    classes.sort()
    class_log_probs_dict = {}

    for i in range(m):
        doc = x_test[i]
        # split doc into list of individual words such that punctuations are kept separate from word
        tokens = word_tokenize(doc)
        # removing punctuations to get final list of words
        tokens = remove_punctution(tokens)
        # removing stop words and doing lemmatization of tokens
        if(clean):
            tokens = clean_data(tokens)
        # if ngram model used
        if(NGRAM):
            Ngram = ngrams(tokens, N)
            if(only_ngram == True):
                tokens = []
            for ngram in Ngram:
                tokens.append(ngram)

        # class_log_probs = sum_log_feature_prob + log_class_prior
        class_log_probs = np.zeros(num_classes)
        for cls in phi.keys():
            log_class_prior = phi[cls]  # values already stored in log form
            sum_log_feature_prob = 0  # np.sum(word_probs_per_class[cls])
            for word in tokens:
                if(word in word_probs_per_class[cls].keys()):
                    sum_log_feature_prob += word_probs_per_class[cls][word]

            class_log_probs[cls-1] = sum_log_feature_prob + log_class_prior
#         print(class_log_probs)
        class_log_probs_dict[i] = class_log_probs
        # due to zero based indexing adding +1
        final_pred = np.argmax(class_log_probs)+1
        preds.append(final_pred)

    return preds, class_log_probs_dict


# ENSEMBLE MODEL (CLASSIFIERS TRAINED ON REVIEW AND SUMMARY)
def ensemble_model_test(class_log_probs_dict, class_log_probs_summary_dict):
    preds = []
    for idx in class_log_probs_dict.keys():
        class_probs = (
            class_log_probs_dict[idx] + class_log_probs_summary_dict[idx])/2
        # due to zero based indexing adding +1
        final_pred = np.argmax(class_probs)+1
        preds.append(final_pred)
    return preds

# BASELINES


def random_baseline(x_test, y_test, phi):
    np.random.seed(0)
    preds = np.random.randint(1, 6, len(y_test))  # upper limit exclusive
    return preds


def majority_baseline(x_test, y_test, phi):
    num_classes = len(phi.keys())
    probs = np.zeros(num_classes+1)
    for key in phi.keys():
        probs[key] = phi[key]
    majority_pred = np.argmax(np.array(probs)[1:])+1
    # print(majority_pred)
    preds = np.full(len(y_test), majority_pred)
    return preds

# UTILITY FUNCTIONS


def accuracy(pred, y):
    m = len(y)
    acc = 0
    for i in range(m):
        if(pred[i] == y[i]):
            acc += 1
    acc /= m
    return acc

# self implementation of confusion matrix


def compute_confusion_matrix(preds, y_test):
    y_test = list(y_test)
    num_classes = len(np.unique(y_test))

    conf_mat = np.zeros((num_classes+1, num_classes+1))

    for i in range(len(y_test)):
        conf_mat[y_test[i]][preds[i]] += 1

    return conf_mat


def print_confusion_matrix(conf_mat):
    print("1, 2, 3, 4, 5 <- predicted class")
    m = len(conf_mat)
    for i in range(1, m):
        print("[", end=' ')
        for j in range(1, m-1):
            print(int(conf_mat[i][j]), end=', ')
        print(int(conf_mat[i][m-1]), "]")


def f1_score(conf_mat, average=None):
    # taken 1 extra to keep 0th row and column empty
    num_classes = conf_mat.shape[0] - 1
    prec_recall_dict = {}
    f1_dict = {}

    for i in range(1, num_classes+1):
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


def plot_confusion_matrix(preds, y_test, part='1_c'):
    y_test = list(y_test)
    conf_mat = confusion_matrix(y_test, preds)
    # print(conf_mat)
    df = pd.DataFrame(conf_mat,
                      index=[1, 2, 3, 4, 5],
                      columns=[1, 2, 3, 4, 5])
    # Plotting the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(df, annot=True, fmt=".1f")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('./Q1/confusion_matrix_test_'+part+'.png')
    plt.show()


def a(x_train, y_train, vocab, class_words_dict, class_num_words_dict, saved=True):
    # Part (a) change saved=True, after calculating params on train data once, to skip recomputing again
    if(saved == False):
        print("Training ...")
    else:
        print("Loading model parameters ...")
    phi,  word_probs_per_class = train(
        x_train, y_train,  vocab, class_words_dict, class_num_words_dict, saved=saved)

    if(saved == False):
        print("Model is now trained.")

    if(argv[2] == 'a'):
        mod_v = len(vocab)
        preds, class_log_probs_dict = predict(
            x_train, y_train, phi, word_probs_per_class, mod_v)
        train_acc = accuracy(preds, y_train)
        print("Accuracy on train data (using review text only) using Multinomial Naive Bayes: ", train_acc)

    return phi, word_probs_per_class


def b(x_test, y_test, phi, word_probs_per_class, mod_v):
    # random baseline
    # print('\n----------------- PART (b) ---------------------------\n')
    if(argv[2] == 'b'):
        preds = random_baseline(x_test, y_test, phi)
        test_acc = accuracy(preds, y_test)
        print("Accuracy on test data using Random baseline: ", test_acc)

        # majority baseline
        preds = majority_baseline(x_test, y_test, phi)
        test_acc = accuracy(preds, y_test)
        print("Accuracy on test data using Majority baseline: ", test_acc)

    # Multinomial Naive Bayes
    preds, class_log_probs_dict = predict(
        x_test, y_test, phi, word_probs_per_class, mod_v)
    test_acc = accuracy(preds, y_test)
    print("Accuracy on test data (using review text only) using Multinomial Naive Bayes: ", test_acc)

    return preds, class_log_probs_dict


def c(preds, y_test):
    # print('\n----------------- PART (c) ---------------------------\n')
    if(argv[2] == 'd'):
        conf_mat = compute_confusion_matrix(preds, y_test)
        return conf_mat

    conf_mat = compute_confusion_matrix(preds, y_test)
    print_confusion_matrix(conf_mat)
    plot_confusion_matrix(preds, y_test)
    return conf_mat


def d(x_train, y_train, x_test, y_test, saved=True):
    # print('\n----------------- PART (d) ---------------------------\n')
    print("Training ...")
    vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
        x_train, y_train, saved=saved, clean=True, part='1_d')
    mod_v = len(vocab)
    # print(mod_v)
    # print("Length of vocabulary after cleaning text data: ", mod_v)
    # retrain
    phi,  word_probs_per_class = train(
        x_train, y_train,  vocab, class_words_dict, class_num_words_dict, saved=saved, part='1_d')
    print("Model is now trained.")
    # test
    print("Testing ...")
    preds_1d, class_log_probs_dict_1d = predict(
        x_test, y_test, phi, word_probs_per_class, mod_v, clean=True)
    test_acc = accuracy(preds_1d, y_test)
    print("Accuracy on test data using Multinomial Naive Bayes after cleaning text: ", test_acc)
    conf_mat = c(preds_1d, y_test)
    print_confusion_matrix(conf_mat)
    plot_confusion_matrix(preds_1d, y_test, '1_d')
    f1_score(conf_mat)
    f1_score(conf_mat, 'macro')


def e(x_train, y_train, x_test, y_test, saved):
    #print('\n----------------- PART (e) ---------------------------\n')
    print("Model using only bigrams as features ... ")
    vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
        x_train, y_train, saved=saved, clean=True, NGRAM=True, N=[2], only_ngram=True, part='1_e_bi')
    mod_v = len(vocab)
    print("Length of vocabulary only bigram: ", mod_v)
    # retrain
    phi,  word_probs_per_class = train(
        x_train, y_train,  vocab, class_words_dict, class_num_words_dict, saved=saved, part='1_e_bi')
    # test
    preds_1d, class_log_probs_dict_1d = predict(
        x_test, y_test, phi, word_probs_per_class, mod_v, clean=True, NGRAM=True, N=[2], only_ngram=True)
    test_acc = accuracy(preds_1d, y_test)
    print("Accuracy on test data using Multinomial Naive Bayes with only bigram features: ", test_acc)

    print("Model using unigrams and bigrams as features ... ")
    vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
        x_train, y_train, saved=saved, clean=True, NGRAM=True, N=[2], part='1_e_uni_bi')
    mod_v = len(vocab)
    print("Length of vocabulary on unigram + bigram: ", mod_v)
    # retrain
    phi,  word_probs_per_class = train(
        x_train, y_train,  vocab, class_words_dict, class_num_words_dict, saved=saved, part='1_e_uni_bi')
    # test
    preds_1d, class_log_probs_dict_1d = predict(
        x_test, y_test, phi, word_probs_per_class, mod_v, clean=True, NGRAM=True, N=[2])
    test_acc = accuracy(preds_1d, y_test)
    print("Accuracy on test data using Multinomial Naive Bayes with unigram + bigram features: ", test_acc)

    print("Model using unigrams, bigrams and trigrams as features ... ")
    vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
        x_train, y_train, saved=saved, clean=True, NGRAM=True, N=[2, 3], part='1_e_uni_bi_tri')
    mod_v = len(vocab)
    print("Length of vocabulary on unigram + bigram + trigram: ", mod_v)
    # retrain
    phi,  word_probs_per_class = train(
        x_train, y_train,  vocab, class_words_dict, class_num_words_dict, saved=saved, part='1_e_uni_bi_tri')
    # test
    preds_1d, class_log_probs_dict_1d = predict(
        x_test, y_test, phi, word_probs_per_class, mod_v, clean=True, NGRAM=True, N=[2, 3])
    test_acc = accuracy(preds_1d, y_test)
    print("Accuracy on test data using Multinomial Naive Bayes with unigram + bigram + trigram  features: ", test_acc)


def f(conf_mat):
    # print('\n----------------- PART (f) ---------------------------\n')
    f1_score(conf_mat)
    f1_score(conf_mat, 'macro')


def g(class_log_probs_dict, saved=True):
    # print('\n----------------- PART (g) ---------------------------\n')
    x_train_sum, y_train_sum, x_test_sum, y_test_sum = summary_data(
        argv[0], argv[1])

    print("Training ...")
    vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
        x_train_sum, y_train_sum, saved=saved, clean=False, part='1_g')
    mod_v = len(vocab)
    #print("Length of vocabulary on original summary text data (without punctuations) : ", mod_v)
    # change saved=True, after calculating params on train data once, to skip recomputing again
    phi,  word_probs_per_class = train(
        x_train_sum, y_train_sum,  vocab, class_words_dict, class_num_words_dict, saved=saved, part='1_g')

    print("Model is now trained ...")
    # Multinomial Naive Bayes - prediction on summary data
    print("Testing")
    preds_summary, class_log_probs_summary_dict = predict(
        x_test_sum, y_test_sum, phi, word_probs_per_class, mod_v, clean=False)

    test_acc = accuracy(preds_summary, y_test_sum)

    print("Accuracy on test data using Multinomial Naive Bayes on Summary Data only: ", test_acc)

    preds_summary_review = ensemble_model_test(
        class_log_probs_dict, class_log_probs_summary_dict)
    test_acc = accuracy(preds_summary_review, y_test_sum)
    print("Accuracy on test data using Multinomial Naive Bayes on Summary Data and Review data ensemble model: ", test_acc)
    plot_confusion_matrix(preds_summary_review, y_test_sum, '1_g')

# MAIN


def main():
    # read data
    # './Q1/reviews_Digital_Music_5.json/Music_Review_train.json'
    train_file_name = argv[0]
    # './Q1/reviews_Digital_Music_5.json/Music_Review_test.json'
    test_file_name = argv[1]
    x_train, y_train, x_test, y_test = read_data(
        train_file_name, test_file_name)

    # print("Length of vocabulary on original text data (without punctuations) : ", mod_v)
    if(argv[2] == 'a'):
        print("Preparing vocabulary ...")
        vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
            x_train, y_train, saved=True)
        mod_v = len(vocab)
        phi, word_probs_per_class = a(
            x_train, y_train, vocab, class_words_dict, class_num_words_dict, saved=True)
        print("Please run part b to get test set accuracy using trained model and other baselines.")
    # Part (b)
    if(argv[2] == 'b'):
        vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
            x_train, y_train, saved=True)
        mod_v = len(vocab)
        phi, word_probs_per_class = a(
            x_train, y_train, vocab, class_words_dict, class_num_words_dict)
        preds, class_log_probs_dict = b(
            x_test, y_test, phi, word_probs_per_class, mod_v)
    # Part (c) - draw confusion matrix of test data
    if(argv[2] == 'c'):
        vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
            x_train, y_train, saved=True)
        mod_v = len(vocab)
        phi, word_probs_per_class = a(
            x_train, y_train, vocab, class_words_dict, class_num_words_dict)
        preds, class_log_probs_dict = b(
            x_test, y_test, phi, word_probs_per_class, mod_v)
        conf_mat = c(preds, y_test)
        f(conf_mat)
    # Part (d) - text cleaning
    if(argv[2] == 'd'):
        d(x_train, y_train, x_test, y_test, saved=True)
    # Part (e)
    if(argv[2] == 'e'):
        # files were too large to save and get a compressed folder within 20MB limit
        e(x_train, y_train, x_test, y_test, saved=False)
    # # Part (f)
    if(argv[2] == 'f'):
        print("Results of part f were combined along with part c (since the original model was the best performing model).")
    # Part (g)
    if(argv[2] == 'g'):
        vocab, class_words_dict, class_num_words_dict = dictionary_prepare(
            x_train, y_train, saved=True)
        mod_v = len(vocab)
        phi, word_probs_per_class = a(
            x_train, y_train, vocab, class_words_dict, class_num_words_dict)
        preds, class_log_probs_dict = b(
            x_test, y_test, phi, word_probs_per_class, mod_v)
        g(class_log_probs_dict, saved=True)


main()

# REFERENCES
# 1. https://machinelearningmastery.com/clean-text-machine-learning-python/
# 2. https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
# 3. https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
# 4. https://www.researchgate.net/publication/337321725_The_Effect_of_Stemming_and_Removal_of_Stopwords_on_the_Accuracy_of_Sentiment_Analysis_on_Indonesian-language_Texts
# 5. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# 6. https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
