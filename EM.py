import numpy as np
from scipy.stats import multivariate_normal


def log_likelihood(likelihood, likelihood_index):
    product = 1
    for i in range(len(likelihood_index)):
        product = product * likelihood[i][likelihood_index[i]]
    return np.log(product)


class EM:

    def __init__(self, max_iter=10, random_seed=26):
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, k, data):
        if len(data) < k:
            print('Number of cluster exceeds size of data')
            return

        # Siyi's Part: Initialize the mean, the variance and the weight for each cluster
        # Comment the following code and implement a better initialization algorithm
        self.mean = np.ndarray(shape=(k, data.shape[1]), dtype=float)
        self.covariance = np.ndarray(shape=(k, data.shape[1], data.shape[1]), dtype=float)
        self.weight = np.ones(shape=k) / k
        clusters = [None] * k
        l = int(data.shape[0] / k)
        for i in range(k-1):
            clusters[i] = data[i*l:(i+1)*l]
        clusters[k-1] = data[(k-1)*l:]
        for i in range(k):
            self.mean[i] = np.mean(clusters[i], axis=0)
            self.covariance[i] = np.cov(np.array(clusters[i]).T)

        # posterior: P(theta | X) likelihood: P(X | theta)
        self.likelihood = np.ndarray(shape=(len(data), k), dtype=float)
        self.posterior = np.ndarray(shape=(k, len(data)), dtype=float)
        old_likelihood = np.ndarray(shape=(len(data), k), dtype=float)
        old_maximum_likelihood_index = np.ndarray(shape=len(data), dtype=int)
        new_maximum_likelihood_index = np.ndarray(shape=len(data), dtype=int)
        for iter in range(self.max_iter):

            likelihood_T = np.ndarray(shape=(k, len(data)), dtype=float)
            for i in range(k):
                likelihood_T[i] = multivariate_normal.pdf(data, mean=self.mean[i], cov=self.covariance[i])
            self.likelihood = likelihood_T.T

            # Expectation
            evidence = np.dot(self.likelihood, self.weight)
            for i in range(k):
                self.posterior[i] = np.multiply(likelihood_T[i], self.weight[i]) / evidence

            # for i in range(k):
            #     for j in range(len(data)):
            #         evidence = np.dot(self.likelihood[j], self.weight)
            #         self.posterior[i][j] = self.likelihood[j][i] * self.weight[i] / evidence
            # Maximization Step
            for i in range(k):
                posterior_sum = np.sum(self.posterior[i])
                self.mean[i] = np.dot(self.posterior[i], data) / posterior_sum
                diff = data - self.mean[i]
                self.covariance[i] = np.zeros(shape=(data.shape[1], data.shape[1]))
                for j in range(len(data)):
                    self.covariance[i] += self.posterior[i][j] * diff[j] * diff[j].reshape(diff[j].shape[0], 1)
                self.covariance[i] = self.covariance[i] / posterior_sum
                if not np.any(self.covariance[i]):
                    self.covariance[i] = np.diag([1e-6] * data.shape[1])
                self.weight[i] = posterior_sum / len(data)

            # log likelihood
            if iter == 0:
                for i in range(len(data)):
                    old_maximum_likelihood_index[i] = np.argmax(self.likelihood[i])
                old_likelihood = self.likelihood.copy()
            else:
                for i in range(len(data)):
                    new_maximum_likelihood_index[i] = np.argmax(self.likelihood[i])
                old_log_likelihood = log_likelihood(old_likelihood, old_maximum_likelihood_index)
                new_log_likelihood = log_likelihood(self.likelihood, new_maximum_likelihood_index)
                if np.abs((new_log_likelihood - old_log_likelihood) / old_log_likelihood) < 0.2:
                    break
                old_likelihood = self.likelihood.copy()
                old_maximum_likelihood_index = new_maximum_likelihood_index.copy()

        clusters = [None] * k
        for i in range(len(data)):
            if clusters[new_maximum_likelihood_index[i]] is None:
                clusters[new_maximum_likelihood_index[i]] = []
            clusters[new_maximum_likelihood_index[i]].append(i)

        return clusters
