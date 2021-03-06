import numpy as np
from scipy.stats import multivariate_normal
from cluster import kMeanCluster
from time import time
reg_cov = 1e-6  # add to the diagonal of covariance for Non-negative regularization.


# judge whether the result converges.
def log_likelihood_sum(likelihood, weights):
    # product = 1
    # for i in range(len(likelihood_index)):
    #     product = product * likelihood[i][likelihood_index[i]]
    # return np.log(product)
    # max_likelihood_arr = likelihood[range(len(likelihood)), likelihood_index]
    probability = (likelihood*weights).sum(axis=1)
    log_prob_sum = np.log(probability).sum()
    return log_prob_sum


def _estimate_gaussian_parameters(data, posterior):
    post_sum = posterior.sum(axis=0) + 10 * np.finfo(posterior.dtype).eps  # weights of component.
    means = np.dot(posterior.T, data) / post_sum[:, np.newaxis]
    covariances = _estimate_gaussian_covariances(posterior, data, post_sum, means)
    post_sum /= data.shape[0]
    return post_sum, means, covariances


def _estimate_gaussian_covariances(posterior, data, nk, means):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = data - means[k]
        covariances[k] = np.dot(posterior[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_cov  # add reg_cov on the diagonal line.
    return covariances


class EM:

    def __init__(self, max_iter=30, threshold=1e-2, random_seed=26, k=2, init_params='kmeans', verbose=False):
        self.max_iter = max_iter
        self.threshold = threshold  # iteration stop threshold
        self.random_seed = random_seed
        self.init_params = init_params
        self.n_components = k
        self.verbose = verbose

    def set_parameters(self, mean, covariance, weight):
        self.means = mean
        self.covariances = covariance
        self.weights = weight

    def initialize(self, data):
        # Siyi's Part: Initialize the mean, the variance and the weight for each cluster
        # Comment the following code and implement a better initialization algorithm
        k = self.n_components
        n_samples = data.shape[0]
        if self.init_params == 'kmeans':
            res = np.zeros((n_samples, self.n_components))
            label = kMeanCluster(num_clusters=self.n_components, ).fit(data).label_
            res[np.arange(n_samples), label] = 1
            weights, means, covariances = _estimate_gaussian_parameters(data, res)

        elif self.init_params == 'random':
            means = np.ndarray(shape=(k, data.shape[1]), dtype=float)
            covariances = np.ndarray(shape=(k, data.shape[1], data.shape[1]), dtype=float)
            weights = np.ones(shape=k) / k

            clusters = [None] * k
            l = int(data.shape[0] / k)
            for i in range(k - 1):
                clusters[i] = data[i * l:(i + 1) * l]
            clusters[k - 1] = data[(k - 1) * l:]
            for i in range(k):
                means[i] = np.mean(clusters[i], axis=0)
                covariances[i] = np.cov(np.array(clusters[i]).T)
        else:
            print('set correct initial params')
            return

        self.set_parameters(means, covariances, weights)

    def print_verbose(self, iter, change, end=False):
        if self.verbose:
            if iter % 10 == 0 or end:
                print('%d iter\t time:%.1f\t change:%.2f' % (iter, time() - self.start_time, change))

    def fit(self, data):
        self.start_time = time()
        k = self.n_components
        n_samples = data.shape[0]
        if len(data) < k:
            print('Number of cluster exceeds size of data')
            return

        self.initialize(data)
        # posterior: P(theta | X) likelihood: P(X | theta)
        self.likelihood = np.ndarray(shape=(len(data), k), dtype=float)
        self.posterior = np.ndarray(shape=(k, len(data)), dtype=float)
        # old_likelihood = np.ndarray(shape=(len(data), k), dtype=float)
        # old_maximum_likelihood_index = np.ndarray(shape=data.shape[0], dtype=int)
        # new_maximum_likelihood_index = np.ndarray(shape=data.shape[0], dtype=int)

        previous_likelihood_sum = 0
        for iter in range(self.max_iter):
            likelihood_T = np.ndarray(shape=(k, len(data)), dtype=float)
            for i in range(k):
                likelihood_T[i] = multivariate_normal.pdf(data, mean=self.means[i], cov=self.covariances[i])
            self.likelihood = likelihood_T.T

            # Expectation
            evidence = np.dot(self.likelihood, self.weights)  # size n of p(x_i).
            for i in range(k):
                self.posterior[i] = np.multiply(likelihood_T[i], self.weights[i]) / evidence
            # for i in range(k):
            #     for j in range(len(data)):
            #         evidence = np.dot(self.likelihood[j], self.weight)
            #         self.posterior[i][j] = self.likelihood[j][i] * self.weight[i] / evidence

            # Maximization Step
            # for i in range(k):
            #     posterior_sum = np.sum(self.posterior[i])
            #     self.means[i] = np.dot(self.posterior[i], data) / posterior_sum
            #     diff = data - self.means[i]
            #     self.covariances[i] = np.zeros(shape=(data.shape[1], data.shape[1])) # ??
            #     for j in range(len(data)):
            #         self.covariances[i] += self.posterior[i][j] * diff[j] * diff[j].reshape(diff[j].shape[0], 1)
            #     self.covariances[i] = self.covariances[i] / posterior_sum
            #     # in case of singular matrix
            #     if not np.any(self.covariances[i]):
            #         self.covariances[i] = np.diag([1e-6] * data.shape[1])
            #     self.weights[i] = posterior_sum / len(data)

            weights, means, covariances = _estimate_gaussian_parameters(data, self.posterior.T)
            self.set_parameters(means, covariances, weights)

            # log likelihood
            if iter == 0:
                previous_likelihood_sum = log_likelihood_sum(self.likelihood, weights)
            else:
                current_likelihood_sum = log_likelihood_sum(self.likelihood, weights)
                change = abs(current_likelihood_sum - previous_likelihood_sum)
                # if abs(current_likelihood_sum - previous_likelihood_sum)  < 0.2:
                previous_likelihood_sum = current_likelihood_sum
                if change <= self.threshold:
                    self.print_verbose(iter, change, end=True)
                    break
                self.print_verbose(iter, change)
        #     else:
        #         for i in range(len(data)):
        #             new_maximum_likelihood_index[i] = np.argmax(self.likelihood[i])
        #         old_log_likelihood = log_likelihood(old_likelihood, old_maximum_likelihood_index)
        #         new_log_likelihood = log_likelihood(self.likelihood, new_maximum_likelihood_index)
        #         if np.abs((new_log_likelihood - old_log_likelihood) / old_log_likelihood) < 0.2:
        #             print('stop update. iteration %i' % iter)
        #             break
        #         old_likelihood = self.likelihood.copy()
        #         old_maximum_likelihood_index = new_maximum_likelihood_index.copy()
        # clusters = [None] * k
        # for i in range(len(data)):
        #     if clusters[new_maximum_likelihood_index[i]] is None:
        #         clusters[new_maximum_likelihood_index[i]] = []
        #     clusters[new_maximum_likelihood_index[i]].append(i)

        new_maximum_likelihood_index = np.argmax(self.likelihood, axis=1)
        return new_maximum_likelihood_index
