"""
logistic_regression.py
----------------------

A module to implement logistic regression
using stochastic gradient descent and the early stop method.
A momentum constant is used as well in the update rule to avoid local minima.

L2-regularization can be added to the update rule calculations
by setting the parameter `reg_const` to a non-zero value.
However, initial experiments have shown that this doesn't improve performance
on the given dataset for this particular homework assignment, and therefore
the constant is set to 0.
"""

from __future__ import division
import numpy as np

# learning_rate = 20 if momentum_constant == 0
learning_rate = 1
epochs_per_validation = 10
convergence_critera = -1e-03 * 5

# regularization constant
# currently regularization is not used as it decreases performance
# set reg_const to around 1e-04 for reasonable results
reg_const = 0
# momentum constant
momentum_constant = 0.9


class LogReg(object):
    """
    A class to perform logistic regression
    using stochastic gradient descent with a momentum term and the early stop method,
    with options to use l2-regularization.

    The early stop method is used to detect convergence and prevent overfitting.
    One training set and a validation set is to be provided and the early stop method
    detects convergence by measuring the difference of the average loss function value
    calculated on the validation set after a set number of epochs of training on the training set.
    """
    def __init__(self, num_weights, weights=None):
        self.num_weights = num_weights
        # initialize weights randomly if none is provided
        self.weights = weights or np.random.randn(num_weights, 1)
        # momentum term
        self.delta_w_old = np.zeros((num_weights, 1))

    def feedforward_sparse(self, x):
        """
        Returns output of system if `x` is the input vector.
        Optimized for sparse input, hence x is in dictionary format: {position : value}

        Example of x:
            feature vector [2, 0, 0, 0, 7, 19]
            would be represented as {0: 2, 4: 7, 5: 19}
        """
        z = 0
        for feature_id, feature_value in x.iteritems():
            z += self.weights[feature_id, 0] * feature_value
        return 1.0/(1.0 + np.exp(-z))

    def sgd(self, (training_data, targets), (validation_data, validation_targets),
            eta=learning_rate):
        """
        Stochastich gradient descent

        Parameters
        ----------
        training_data : list
            list of dicts that represent sparse vectors.
            Example of one such dict:
                feature vector [2, 0, 0, 0, 7, 19]
                would be represented as {0: 2, 4: 7, 5: 19}
        targets : list (column vector)
            a column vector containing the target values of each training sample.
            Example: [[0], [0], [1], [1]]
        (validation_data, validation_targets) : (list, list (column vector))
            same format as `training_data` and `targets`
        eta : float
            the learning rate

        Returns
        -------
        A column vector.
        The weights of the system from the previous convergence check
        after convergence was detected.
        """
        converged = False
        epoch = 0

        # calculate the average error for the validation set
        print "Calculating the loss function value on the validation set ..."
        loss_old = self.get_avg_loss((validation_data, validation_targets))
        print "Epoch %d: average loss function value on validation set: %f" \
              % (epoch, loss_old)
        print "Convergence critera: when difference > %f" % (convergence_critera)
        weights_old = np.copy(self.weights)

        while not converged:
            epoch += 1
            # go through the training data
            for i, x in enumerate(training_data):
                h_i = self.feedforward_sparse(x)
                error_term_i = targets[i, 0] - h_i
                delta_w_i = np.zeros(self.weights.shape)
                # gradient descent on weights
                # sparse vector addition optimization
                for feature_id, feature_value in x.iteritems():
                    delta_w_i[feature_id, 0] = eta * (
                        error_term_i * feature_value
                    )

                # regularization
                if reg_const != 0:
                    delta_w_i -= reg_const * self.weights
                # momentum term
                if momentum_constant != 0:
                    delta_w_i += momentum_constant * self.delta_w_old
                # gradient descent
                self.weights += delta_w_i
                self.delta_w_old = delta_w_i

            # calculate the average error for the validation set and check for convergence
            if epoch % epochs_per_validation == 0:
                loss_new = self.get_avg_loss(
                    (validation_data, validation_targets))
                difference = loss_new - loss_old
                print "Epoch %d: average loss function value: %f  |  Difference: %f" \
                      % (epoch, loss_new, difference)

                if difference > convergence_critera:
                    converged = True
                else:
                    # not converged, keep training
                    # this rounds data will be next round's old data
                    weights_old = np.copy(self.weights)
                    loss_old = loss_new

        self.weights = weights_old
        return self.weights

    def get_avg_loss(self, (validation_data, validation_targets)):
        """
        Calculates the average ordinary least square cost of the system
        on a validation set.

        Parameters
        ----------
        validation_data : list
            list of dicts that represent sparse vectors.
            Example of one such dict:
                feature vector [2, 0, 0, 0, 7, 19]
                would be represented as {0: 2, 4: 7, 5: 19}
        validation_targets : list (column vector)
            a column vector containing the target values of each training sample.
            Example: [[0], [0], [1], [1]]
        """
        validation_size = len(validation_data)
        loss_sum = 0
        for i, x in enumerate(validation_data):
            h_i = self.feedforward_sparse(x)
            loss_sum += (h_i - validation_targets[i, 0])**2
        avg_loss = 0.5 * loss_sum / validation_size
        # Regularization removed from loss calculation
        # to be able to compare results between runs with and without regularization
        # and use the same convergence criteria.
        # Uncomment below to add it back
        # l2_term = reduce(lambda acc, x: acc + x[0]**2, self.weights)[0]
        # if reg_const != 0:
        #     l2_term = reduce(lambda acc, x: acc + x[0]**2, self.weights)[0]
        #     avg_loss += 0.5 * reg_const * l2_term
        return avg_loss

    def get_confusion_matrix(self, inputs, targets):
        """
        Parameters
        ----------
        inputs : list
            list of dicts that represent sparse vectors.
            Example of one such dict:
                feature vector [2, 0, 0, 0, 7, 19]
                would be represented as {0: 2, 4: 7, 5: 19}
        targets : list (column vector)
            a column vector containing the target values of each training sample.
            Example: [[0], [0], [1], [1]]

        Returns
        -------
        A 2d confusion matrix in the format:
        [[ tp,  fp],
         [ fn,  tn]]

        where:
            tp: true positive
            fp: false positive
            fn: false negative
            tn: true negative
        """
        confusion_matrix = np.zeros((2, 2))
        for i, x in enumerate(inputs):
            h_i = self.feedforward_sparse(x)
            h_i = 0 if h_i < 0.5 else 1
            actual_i = targets[i, 0]

            if h_i == 1 and actual_i == 1:
                # True positive
                confusion_matrix[0, 0] += 1
            elif h_i == 1 and actual_i == 0:
                # False positive
                confusion_matrix[0, 1] += 1
            elif h_i == 0 and actual_i == 1:
                # False negative
                confusion_matrix[1, 0] += 1
            elif h_i == 0 and actual_i == 0:
                # True negative
                confusion_matrix[1, 1] += 1
        return confusion_matrix
