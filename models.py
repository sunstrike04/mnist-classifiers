import numpy as np
import logging
from utils import compute_accuracy

class NearestNeighbor(object):
    def __init__(self, data, labels, k):
        """
        Args:
            data: n x d matrix with a d-dimensional feature for each of the n
            points
            labels: n x 1 vector with the label for each of the n points
            k: number of nearest neighbors to use for prediction
        """
        self.k = k
        self.data = data
        self.labels = labels

    def train(self):
        """_
        Trains the model and stores in class variables whatever is necessary to
        make predictions later.
        """
        # BEGIN YOUR CODE
        # For k-NN, training is just memorizing the data.
        # The data is already stored in self.data and self.labels.
        # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
        Returns:
            y: n vector with the predicted label for each of the n points
        """
        # BEGIN YOUR CODE
        num_test = x.shape[0]
        num_train = self.data.shape[0]
        dists = np.zeros((num_test, num_train))

        # Using the identity: ||a-b||^2 = a^2 + b^2 - 2a*b
        # This avoids large intermediate matrices and is memory efficient.
        sum_x_sq = np.sum(x**2, axis=1, keepdims=True)
        sum_data_sq = np.sum(self.data**2, axis=1)
        dot_product = np.dot(x, self.data.T)
        dists = np.sqrt(sum_x_sq + sum_data_sq - 2 * dot_product)

        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Get the k nearest neighbors' labels
            closest_y_indices = np.argsort(dists[i, :])[:self.k]
            closest_y = self.labels[closest_y_indices]
            # Predict the most common label
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred
        # END YOUR CODE

    def get_nearest_neighbors(self, x, k):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
            k: number of nearest neighbors to return
        Returns:
            top_imgs: n x k x d vector containing the nearest neighbors in the
            training data, top_imgs must be sorted by the distance to the
            corresponding point in x.
        """
        # BEGIN YOUR CODE
        num_test = x.shape[0]
        # Using the identity: ||a-b||^2 = a^2 + b^2 - 2a*b
        sum_x_sq = np.sum(x**2, axis=1, keepdims=True)
        sum_data_sq = np.sum(self.data**2, axis=1)
        dot_product = np.dot(x, self.data.T)
        dists = np.sqrt(sum_x_sq + sum_data_sq - 2 * dot_product)

        top_k_indices = np.argsort(dists, axis=1)[:, :k]
        top_imgs = self.data[top_k_indices]
        return top_imgs
        # END YOUR CODE


class LinearClassifier(object):
    def __init__(self, data, labels, epochs=10, lr=1e-3, reg_wt=3e-5, writer=None):
        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.lr = lr
        self.reg_wt = reg_wt
        self.rng = np.random.RandomState(1234)
        std = 1. / np.sqrt(data.shape[1])
        self.w = self.rng.uniform(-std, std, size=(self.data.shape[1], 10))
        self.writer = writer

    def compute_loss_and_gradient(self):
        """
        Computes total loss and gradient of total loss with respect to weights
        w.  You may want to use the `data, w, labels, reg_wt` attributes in
        this function.
        
        Returns:
            data_loss, reg_loss, total_loss: 3 scalars that represent the
                losses $L_d$, $L_r$ and $L$ as defined in the README.
            grad_w: d x 10. The gradient of the total loss (including
            the regularization term), wrt the weight.
        """
        # BEGIN YOUR CODE
        num_train = self.data.shape[0]

        # Forward pass: compute scores and probabilities
        scores = self.data.dot(self.w) # (N, C)

        # Numerically stable softmax
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, C)

        # Compute data loss (cross-entropy)
        correct_log_probs = -np.log(probs[range(num_train), self.labels])
        data_loss = np.sum(correct_log_probs) / num_train

        # Compute regularization loss
        reg_loss = 0.5 * self.reg_wt * np.sum(self.w * self.w)

        # Total loss
        total_loss = data_loss + reg_loss

        # Compute gradient
        dscores = probs
        dscores[range(num_train), self.labels] -= 1
        dscores /= num_train

        grad_w = self.data.T.dot(dscores)
        grad_w += self.reg_wt * self.w # Add regularization gradient

        return data_loss, reg_loss, total_loss, grad_w
        # END YOUR CODE

    def train(self):
        """Train the linear classifier using gradient descent"""
        for i in range(self.epochs):
            # BEGIN YOUR CODE
            data_loss, reg_loss, total_loss, grad_w = self.compute_loss_and_gradient()

            # Update weights
            self.w -= self.lr * grad_w

            if self.writer is not None and i % 100 == 0:
                self.writer.add_scalar('total_loss', total_loss, i)
                self.writer.add_scalar('data_loss', data_loss, i)
                self.writer.add_scalar('reg_loss', reg_loss, i)
            # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
        Returns:
            y: n vector with the predicted label for each of the n points
        """
        # BEGIN YOUR CODE
        scores = x.dot(self.w)
        return np.argmax(scores, axis=1)
        # END YOUR CODE
