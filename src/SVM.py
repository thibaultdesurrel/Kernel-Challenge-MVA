import numpy as np
import cvxopt
from cvxopt import matrix


class SVM:
    def __init__(self, C, kernel, thresh=1e-3):
        """
        Parameters :
        C : float, regularization paramter
        thresh : float, minimum threshold for the prediction
        """
        self.C = C
        self.kernel = kernel
        self.threshold = thresh
        self.X = None
        self.y = None
        self.alpha = None
        self.support_vectors_indices = None
        self.support_vectors = None

    def fit(self, X, y, K, class_weights):
        """
        Parameters :
        X : array of training data (nb_samples x nb_features)
        y : list of training labels in {-1,1} (nb_samples)
        K : precomputed Gram matrix of X using kernel (nb_samples x nb_samples)
        class_weights : array of the weight for each sample

        Returns :
        alpha : nb_samples array
        """
        self.X = X
        self.y = y

        nb_samples = len(X)

        # We start by defining the different quantities we need in order to solve our problem

        P = matrix(K)
        q = matrix(-y.astype("float"))
        G = matrix(
            np.block(
                [
                    [np.diag(np.squeeze(y).astype("float"))],
                    [-np.diag(np.squeeze(y).astype("float"))],
                ]
            )
        )
        h = np.concatenate((self.C * np.ones(nb_samples), np.zeros(nb_samples)))
        h[:nb_samples] *= (y == 1) * class_weights[0] + (y == -1) * class_weights[1]
        h = matrix(h)

        # Solve the problem using cvxopt
        solver = cvxopt.solvers.qp(P=P, q=q, G=G, h=h)
        solution = solver["x"]
        self.alpha = np.squeeze(np.array(solution))

        # Retrieve the support vectors
        self.support_vectors_indices = (
            np.squeeze(np.abs(np.array(solution))) > self.threshold
        )
        self.alpha = self.alpha[self.support_vectors_indices]
        self.support_vectors = self.X[self.support_vectors_indices]

    def predict_logit(self, X):
        """
        Parameters :
        X : array of data for which we want to predict the label (nb_samples x nb_features)

        Returns :
        Array of the predicted logits (n_samples)
        """
        K = self.kernel(X, self.support_vectors)
        return K @ self.alpha

    def predict_class(self, X):
        """
        Parameters :
        X : array of data for which we want to predict the label (nb_samples x nb_features)

        Returns :
        Array of the predicted labels (n_samples)
        """
        K = self.kernel(X, self.support_vectors)
        y = np.dot(K, self.alpha)
        return np.where(y > self.threshold, 1, -1)
