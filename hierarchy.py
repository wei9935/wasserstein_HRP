import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from wasserstein_distance import sliced_wasserstein
from scipy.cluster.hierarchy import dendrogram, linkage

class wasserstein_HC:
    def __init__(self, data, method='ward'):
        self.labels = data.index
        self.data = np.array(data)
        self.method = method
        self.n = data.shape[0]
        self.linkage_matrix = []
        self.clusters = {i: [i] for i in range(self.n)}
        self.distances = self._sliced_wasserstein_distances()

    def _sliced_wasserstein_distances(self):
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                distances[i, j] = self._pair_sliced_wasserstein(self.data[i], self.data[j])
                distances[j, i] = distances[i, j]
        return distances

    def _pair_sliced_wasserstein(self, a1, a2, centering=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if len(a1.shape) == 1:
            a1 = a1.reshape(-1, 1)
        if len(a2.shape)==1:
            a2 = a2.reshape(-1, 1)
        a1 = torch.tensor(a1, dtype=torch.float, device=device)
        a2 = torch.tensor(a2, dtype=torch.float, device=device)
        d = a1.shape[1]
        if centering:
            mu1 = torch.mean(a1, dim=0)
            mu2 = torch.mean(a2, dim=0)
            a1 = a1 - mu1
            a2 = a2 - mu2
        m2_Xc = torch.mean(torch.linalg.norm(a1, dim=1) ** 2) / d
        m2_Yc = torch.mean(torch.linalg.norm(a2, dim=1) ** 2) / d
        sw = torch.abs(m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2))
        return float(sw)

    def _compute_cluster_distance(self, cluster1, cluster2):
        if self.method == 'single':
            return self._single_linkage_distance(cluster1, cluster2)

        elif self.method == 'complete':
            return self._complete_linkage_distance(cluster1, cluster2)

        elif self.method == 'average':
            return self._average_linkage_distance(cluster1, cluster2)

        elif self.method == 'ward':
            return self._ward_distance(cluster1, cluster2)
        
        elif self.method == 'wass':
            return self.cluster_wasserstein(self.data[cluster1], self.data[cluster2])

        else:
            raise ValueError(f"Method '{self.method}' not supported.")

    def _single_linkage_distance(self, cluster1, cluster2):
        min_dist = np.inf
        for p1 in cluster1:
            for p2 in cluster2:
                dist = self.distances[p1, p2]
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _complete_linkage_distance(self, cluster1, cluster2):
        max_dist = -np.inf
        for p1 in cluster1:
            for p2 in cluster2:
                dist = self.distances[p1, p2]
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def _average_linkage_distance(self, cluster1, cluster2):
        total_dist = 0
        count = 0
        for p1 in cluster1:
            for p2 in cluster2:
                total_dist += self.distances[p1, p2]
                count += 1
        return total_dist / count

    def _ward_distance(self, clust1, clust2):
        mean1 = np.mean(self.data[clust1], axis=0)
        mean2 = np.mean(self.data[clust2], axis=0)
        combined = np.vstack([self.data[clust1], self.data[clust2]])
        mean_combined = np.mean(combined, axis=0)
        dist = len(clust1) * np.sum((mean1 - mean_combined) ** 2) + len(clust2) * np.sum((mean2 - mean_combined) ** 2)
        return dist

    def cluster_wasserstein(self, clust1, clust2):
        w_dist = self._pair_sliced_wasserstein(self.data[[clust1]], self.data[[clust2]])
        return w_dist

    def fit(self):
        while len(self.clusters) > 1:
            closest_clusters = None
            min_dist = np.inf
            cluster_keys = list(self.clusters.keys())
            for i in range(len(cluster_keys) - 1):
                for j in range(i + 1, len(cluster_keys)):
                    c1, c2 = cluster_keys[i], cluster_keys[j]
                    dist = self._compute_cluster_distance(self.clusters[c1], self.clusters[c2])
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusters = (c1, c2)

            c1, c2 = closest_clusters
            new_cluster = self.clusters[c1] + self.clusters[c2]
            new_cluster_idx = max(self.clusters.keys()) + 1
            self.clusters[new_cluster_idx] = new_cluster
            del self.clusters[c1]
            del self.clusters[c2]

            self.linkage_matrix.append([c1, c2, min_dist, len(new_cluster)])
        self.linkage_matrix = np.array(self.linkage_matrix)
        #return np.array(self.linkage_matrix)

    def plt_dendrogram(self, save_path=None):
        plt.figure(figsize=(10, 5))
        dendrogram(self.linkage_matrix, labels=self.labels)
        plt.title('Hierarchical Clustering Dendrogram with Wassestein Distance'), plt.xlabel('Assets')
        plt.xticks(rotation=45), plt.yticks([])
        plt.tick_params(axis='y', which='both', left=False)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

class HierarchicalClutering:
    def __init__(self, data, method='ward'):
        self.labels = data.index
        self.data = np.array(data)
        self.method = method
        self.n = data.shape[0]
        self.linkage_matrix = []
        self.clusters = {i: [i] for i in range(self.n)}
        self.distances = self._compute_initial_distances()

    def _compute_initial_distances(self):
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                distances[i, j] = self._euclidean_distance(self.data[i], self.data[j])
                distances[j, i] = distances[i, j]
        return distances

    def _euclidean_distance(self, p1, p2):
        return float(np.sqrt(np.sum((p1 - p2) ** 2)))


    def _compute_cluster_distance(self, cluster1, cluster2):
        if self.method == 'single':
            return self._single_linkage_distance(cluster1, cluster2)

        elif self.method == 'complete':
            return self._complete_linkage_distance(cluster1, cluster2)

        elif self.method == 'average':
            return self._average_linkage_distance(cluster1, cluster2)

        elif self.method == 'ward':
            return self._ward_distance(cluster1, cluster2)

        else:
            raise ValueError(f"Method '{self.method}' not supported.")

    def _single_linkage_distance(self, cluster1, cluster2):
        min_dist = np.inf
        for p1 in cluster1:
            for p2 in cluster2:
                dist = self.distances[p1, p2]
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _complete_linkage_distance(self, cluster1, cluster2):
        max_dist = -np.inf
        for p1 in cluster1:
            for p2 in cluster2:
                dist = self.distances[p1, p2]
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def _average_linkage_distance(self, cluster1, cluster2):
        total_dist = 0
        count = 0
        for p1 in cluster1:
            for p2 in cluster2:
                total_dist += self.distances[p1, p2]
                count += 1
        return total_dist / count

    def _ward_distance(self, clust1, clust2):
        mean1 = np.mean(self.data[clust1], axis=0)
        mean2 = np.mean(self.data[clust2], axis=0)
        combined = np.vstack([self.data[clust1], self.data[clust2]])
        mean_combined = np.mean(combined, axis=0)
        dist = len(clust1) * np.sum((mean1 - mean_combined) ** 2) + len(clust2) * np.sum((mean2 - mean_combined) ** 2)
        return dist


    def fit(self):
        while len(self.clusters) > 1:
            closest_clusters = None
            min_dist = np.inf
            cluster_keys = list(self.clusters.keys())
            for i in range(len(cluster_keys) - 1):
                for j in range(i + 1, len(cluster_keys)):
                    c1, c2 = cluster_keys[i], cluster_keys[j]
                    dist = self._compute_cluster_distance(self.clusters[c1], self.clusters[c2])
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusters = (c1, c2)

            c1, c2 = closest_clusters
            new_cluster = self.clusters[c1] + self.clusters[c2]
            new_cluster_idx = max(self.clusters.keys()) + 1
            self.clusters[new_cluster_idx] = new_cluster
            del self.clusters[c1]
            del self.clusters[c2]

            self.linkage_matrix.append([c1, c2, min_dist, len(new_cluster)])
        self.linkage_matrix = np.array(self.linkage_matrix)
        #return np.array(self.linkage_matrix)

    def plt_dendrogram(self, save_path=None):
        plt.figure(figsize=(10, 5))
        dendrogram(self.linkage_matrix, labels=self.labels)
        plt.title('Hierarchical Clustering Dendrogram with Euclidean Distance'), plt.xlabel('Assets')
        plt.xticks(rotation=45), plt.yticks([])
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()



def inverse_variance_weights(cov_matrix):
    ivp = 1.0 / np.diag(cov_matrix)
    ivp /= ivp.sum()
    return ivp

def cluster_var(cov_matrix, cluster_items):
    cov_slice = cov_matrix.loc[cluster_items, cluster_items]
    weights = inverse_variance_weights(cov_slice).reshape(-1, 1)
    cluster_var = np.dot(weights.T, np.dot(cov_slice, weights))[0, 0]
    return cluster_var

def quasi_diag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

def rec_bipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[int(j):int(k)] for i in cItems for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = cluster_var(cov, cItems0)
            cVar1 = cluster_var(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w

def hierarchical_risk_parity(cov, linkage_matrix):
    sort_ix = quasi_diag(linkage_matrix)
    sort_ix = [cov.columns[i] for i in sort_ix]
    w = rec_bipart(cov, sort_ix)
    return w



