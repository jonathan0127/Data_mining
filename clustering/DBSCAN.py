import numpy as np

class DBSCAN:
    def __init__(self, train_data=None, test_data=None, eps=0.5, min_samples=5):
        self.train_data = np.array(train_data) if train_data is not None else None
        self.test_data = np.array(test_data) if test_data is not None else None
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, data=None):
        if data is not None:
            self.train_data = np.array(data)
        return self.fit()

    def getNeighbors(self, idx):
        dists = np.linalg.norm(self.train_data - self.train_data[idx], axis=1)
        return np.where(dists <= self.eps)[0]

    def fit(self):
        n = self.train_data.shape[0]
        labels = np.full(n, -1, dtype=int)  # -1 means noise
        cluster_idx = 0
        used = np.zeros(n, dtype=bool)

        for i in range(n):
            if used[i]: continue
            used[i] = True
            neighbors = self.getNeighbors(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise
            else:
                labels[i] = cluster_idx
                neighbors = list(neighbors[neighbors != i])

                while neighbors:
                    j = neighbors.pop()

                    if not used[j]:
                        used[j] = True
                        j_neighbors = self.getNeighbors(j)
                        if len(j_neighbors) >= self.min_samples: # 如果長度太小表示 j 是噪點
                            for jid in j_neighbors:
                                if jid not in neighbors and labels[jid] == -1:
                                    neighbors.append(jid)

                    if labels[j] == -1:
                        labels[j] = cluster_idx
                cluster_idx += 1
        return labels