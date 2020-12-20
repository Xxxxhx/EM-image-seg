from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
import numpy as np


class kMeanCluster:
    def __init__(self, num_clusters, kpp_initial='True'):
        self.num_clusters = num_clusters
        self.kpp_initial = kpp_initial

    def kmeanpp_initial(self):
        # k-means ++ initialization for remaining num_cluster-1 centroid
        n_candidates = int(self.num_clusters / 2)
        # optimal pre-computed for 'euclidean_distances'
        x_squared_norms = row_norms(self.data, squared=True)
        min_dist_sqr = euclidean_distances(
            self.init_centr[0, np.newaxis], self.data, Y_norm_squared=x_squared_norms,
            squared=True)
        dis_sum = min_dist_sqr.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, self.num_clusters):
            # roulette wheel selection to pick n_candidates.
            # choose the optimal one as the next centroid.
            rand_vals = np.random.random_sample(n_candidates) * dis_sum
            candidate_cent_indexs = np.searchsorted(np.cumsum(min_dist_sqr),  # accumulative sum
                                                    rand_vals)
            # Compute distances to the selected candidates
            distances_to_candidates = euclidean_distances(
                self.data[candidate_cent_indexs], self.data, Y_norm_squared=x_squared_norms, squared=True)

            # update closest distances squared and potential for each candidate
            np.minimum(min_dist_sqr, distances_to_candidates,
                       out=distances_to_candidates)
            candidates_dist_sums = distances_to_candidates.sum(axis=1)

            # Decide which candidate is the optimal
            best_candidate = np.argmin(candidates_dist_sums)
            dis_sum = candidates_dist_sums[best_candidate]
            min_dist_sqr = distances_to_candidates[best_candidate]
            best_candidate = candidate_cent_indexs[best_candidate]
            self.init_centr[c] = self.data[best_candidate]

    def predict(self, data):
        x_squared_norms = row_norms(data, squared=True)
        dis = euclidean_distances(
            self.centroids, data, Y_norm_squared=x_squared_norms)
        # dis.shape
        nearest_cent_ind = np.argmin(dis, axis=0)

        # predict which centroid they belong to.
        return nearest_cent_ind

    def fit(self, data, max_iteration=5):
        self.data = data
        # random init first centroid
        # pick up a random sample point as the 1st cent.
        initial_inx = np.random.randint(data.shape[0], size=self.num_clusters)
        self.init_centr = self.data[initial_inx].copy()

        if self.kpp_initial:
            self.kmeanpp_initial()

        # update iteration
        max_itr = max_iteration
        self.centroids = self.init_centr.copy()
        x_squared_norms = row_norms(self.data, squared=True)

        for itr in range(max_itr):
            dis = euclidean_distances(
                self.centroids, self.data, Y_norm_squared=x_squared_norms)
            nearest_cent_ind = np.argmin(dis, axis=0)

            self.inertia = np.min(dis, axis=0).sum()

            stable = 0
            for i in range(self.num_clusters):
                # iteratively select a cluster of points and calculate their center
                cluster = self.data[nearest_cent_ind == i]
                if cluster.shape[0] != 0:
                    new_ct = np.sum(cluster, axis=0) / cluster.shape[0]
                    if (self.centroids[i] == new_ct).all():
                        stable += 1
                    else:
                        self.centroids[i] = new_ct
            self.label_ = nearest_cent_ind
            if stable == self.num_clusters:
                # print('no update. takes %i iterations' % itr)
                break

        return self
