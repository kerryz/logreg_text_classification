from __future__ import division
import numpy as np
# TODO: regularization, momentum term

learning_rate = 1e04 * 2  # TODO: normalize my data...
reg_const = 1  # regularization constant
# momentum_constant = 0.9
epochs_per_validation = 10
convergence_critera = -1e-03 * 5


class LogReg(object):
    """docstring for LogReg"""
    def __init__(self, num_weights, weights=None):
        self.num_weights = num_weights
        # initialize weights randomly if none is provided
        self.weights = weights or np.random.randn(num_weights, 1)

    def feedforward_sparse(self, x):
        """
        Returns output of system if `x` is the input vector.
        Optimized for sparse input, hence x is in dictionary format: {position : value}

        Example of x:
            [2, 0, 0, 0, 7, 19] would be represented as {0: 2, 4: 7, 5: 19}
        """
        z = 0
        for feature_id, feature_value in x.iteritems():
            z += self.weights[feature_id, 0] * feature_value
        return 1.0/(1.0 + np.exp(-z))

    def sgd(self, (training_data, targets), (validation_data, validation_targets),
            eta=learning_rate, mini_batch_size=1):
        """Stochastich gradient descent"""
        converged = False
        epoch = 0

        # calculate the average error for the validation set
        print "Calculating the squared error on the validation set ..."
        squared_error_old = self.get_avg_squared_error((validation_data, validation_targets))
        print "Epoch %d: average squared error on validation set: %f" % (epoch, squared_error_old)
        print "Convergence critera: when difference > %f" % (convergence_critera)
        weights_old = self.weights

        while not converged:
            epoch += 1

            weights_old = self.weights
            # go through the training data
            for i, x in enumerate(training_data):
                h_i = self.feedforward_sparse(x)
                error_term = targets[i, 0] - h_i
                # sparse vector addition optimization
                for feature_id, feature_value in x.iteritems():
                    # gradient descent
                    self.weights[feature_id, 0] += eta * error_term * feature_value

            # calculate the average error for the validation set
            if epoch % epochs_per_validation == 0:
                squared_error_new = self.get_avg_squared_error(
                    (validation_data, validation_targets))
                difference = squared_error_new - squared_error_old
                print "Epoch %d: average squared error on validation set: %f  |  Difference: %f" \
                      % (epoch, squared_error_new, difference)

                if difference > convergence_critera:
                    converged = True

                squared_error_old = squared_error_new

        self.weights = weights_old
        return weights_old

    def get_avg_squared_error(self, (validation_data, validation_targets)):
        validation_size = len(validation_data)
        squared_error_sum = 0
        for i, x in enumerate(validation_data):
            h_i = self.feedforward_sparse(x)
            squared_error_sum += (h_i - validation_targets[i, 0])**2
        return 0.5 * squared_error_sum / validation_size

    def get_confusion_matrix(self, inputs, targets):
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
