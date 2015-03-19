from __future__ import division
import numpy as np
# TODO: regularization, momentum term

learning_rate = 1e04 * 2  # TODO: normalize my data...
epochs_per_validation = 10
convergence_critera = -1e-03 * 5

# regularization constant
# currently regularization is not used as it decreases performance
# set reg_const to around 1e-07*2 for reasonable results
reg_const = 0
# momentum_constant = 0.9


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
        print "Calculating the loss function value on the validation set ..."
        loss_old = self.get_avg_loss((validation_data, validation_targets))
        print "Epoch %d: average loss function value on validation set: %f" \
              % (epoch, loss_old)
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
                    self.weights[feature_id, 0] += (eta * error_term * feature_value
                                                    - eta * reg_const * self.weights[feature_id, 0])

            # calculate the average error for the validation set
            if epoch % epochs_per_validation == 0:
                loss_new = self.get_avg_loss(
                    (validation_data, validation_targets))
                difference = loss_new - loss_old
                print "Epoch %d: average loss function value: %f  |  Difference: %f" \
                      % (epoch, loss_new, difference)

                if difference > convergence_critera:
                    converged = True

                loss_old = loss_new

        self.weights = weights_old
        return weights_old

    def get_avg_loss(self, (validation_data, validation_targets)):
        validation_size = len(validation_data)
        loss_sum = 0
        for i, x in enumerate(validation_data):
            h_i = self.feedforward_sparse(x)
            loss_sum += (h_i - validation_targets[i, 0])**2
        l2_term = reduce(lambda acc, x: acc + x[0]**2, self.weights)[0]
        return ((0.5*loss_sum) + (0.5*reg_const * l2_term)) / validation_size

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
