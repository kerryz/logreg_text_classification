"""
main.py
-------
Performs logistic regression on given data.
Uses k-fold cross validation for evaluation
and will output the average recall, precision and f1 score to terminal.

The feature files are separated into three sets: a test set, a training set, and a validation set.
The validation set is used to determine convergence
and prevent overfitting during the training phase.
The test set is used to evaluate system's performance after training.

To run
------
python main.py [data_dir]

Command line arguments:
    `data_dir` is the path to the directory containing all the feature files.

Defaults:
    `data_dir` = './data/'

Assumes the following directory structure for `data_dir`
    data_dir
    |-- s1_features
    |-- s2_features
    |-- ...

    Where the feature file contents are of the format:
        <label> <feature1>:<value1> <feature2>:<value2>
            where <label> == 0 or 1 since we're using logistic regression
            example: "0 22199:0.0 13448:0.00148313739872 15346:7.97308993901e-05"
"""

from __future__ import division
import numpy as np
import os, sys
import json

from logistic_regression import LogReg

def_data_dir = './data/'


def main(data_dir=def_data_dir):
    # get all feature file paths
    feature_files = [f for f in os.listdir(data_dir) if f.endswith('_features')]
    feature_file_paths = sorted([os.path.join(data_dir, f) for f in feature_files])

    # calculate vocabulary size
    with open(os.path.join(data_dir, 'word2id.json')) as word2id_json_file:
        vocab_size = len(json.load(word2id_json_file))

    precision_sum, recall_sum, f1_sum = 0, 0, 0

    # setting up k-fold cross validation
    for i, test_set_path in enumerate(feature_file_paths):
        # separate into training, validation, and test sets
        training_set_paths = get_list_without_index(feature_file_paths, i)
        validation_set_path = training_set_paths.pop(i - 1)

        # load training data
        training_data, targets = load_feature_files(training_set_paths)
        # load validation data
        validation_data, validation_targets = load_feature_file(validation_set_path)

        # load the Logistic Regression class
        logreg = LogReg(vocab_size)
        # stochastic gradient descent
        logreg.sgd((training_data, targets), (validation_data, validation_targets))

        # load test data
        test_data, test_targets = load_feature_file(test_set_path)
        # calculate confusion matrix
        confusion_matrix = logreg.get_confusion_matrix(test_data, test_targets)

        # True Positive, False Positive, False Negative, True Negative
        tp = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tn = confusion_matrix[1, 1]

        # calculate precision, recall, and F1 scores
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        # set up set names for printing
        test_set_name = test_set_path.split(os.path.sep)[-1]
        training_set_names = map(lambda path: path.split(os.path.sep)[-1], training_set_paths)
        validation_set_name = validation_set_path.split(os.path.sep)[-1]
        print
        print '-' * 40
        print 'test set        ', test_set_name
        print 'training sets:', training_set_names
        print 'validation set: ', validation_set_name
        print
        print "Confusion matrix"
        print confusion_matrix
        print "Recall:   ", recall
        print "Precision:", precision
        print "F1 Score: ", f1
        print '-' * 40
        print

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    total_iterations = len(feature_file_paths)
    print "=" * 40
    print "Averages of %d-fold cross validation" % total_iterations
    print "Recall:   ", recall_sum / total_iterations
    print "Precision:", precision_sum / total_iterations
    print "F1 Score: ", f1_sum / total_iterations
    print "=" * 40


def load_feature_files(file_paths):
    training_data = []
    for i, file_path in enumerate(file_paths):
        training_data_i, targets_i = load_feature_file(file_path)
        training_data += training_data_i
        # for the first iteration, create `targets`
        if i == 0:
            targets = targets_i
        else:
            targets = np.concatenate((targets, targets_i), axis=0)
    return training_data, targets


def load_feature_file(file_path):
    print "loading feature file", file_path
    targets = []
    training_data = []
    with open(file_path) as infile:
        for line in infile:
            targets.append(float(line[0]))
            features_str_list = line[2:].split(" ")  # ["feature_id:feature_value"]
            # create list in format [(feature_id, feature_value)]
            features_tuple_list = [(f_id, f_val)
                                   for (f_id, f_val)
                                   in map(lambda s: s.split(":"), features_str_list)
                                   ]
            # make a dict out of the list of tuples
            # converting feature_id and feature_value to numbers
            x_i = dict(map(
                lambda (f_id, f_val): (int(f_id), float(f_val)),
                features_tuple_list)
            )
            training_data.append(x_i)
    return training_data, np.transpose(np.array([targets]))


def get_list_without_index(list, index):
    list_without_index = []
    for i, v in enumerate(list):
        if i == index:
            continue
        list_without_index.append(v)
    return list_without_index


if __name__ == "__main__":
    data_dir = def_data_dir

    # use command line arguments if provided
    arg_count = len(sys.argv)
    if arg_count > 1:
        data_dir = sys.argv[1]

    main(data_dir)
