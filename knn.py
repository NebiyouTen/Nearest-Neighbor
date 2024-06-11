import numpy as np

class KNN(object):
    """docstring for KNN."""

    def __init__(self, data, labels, k = 1, normalize=False, norm_type = "min_max" ):
        super(KNN, self).__init__()
        self.k = k
        self.data = data.copy()
        self.labels = labels.reshape(-1)

        dr = 1 / len(self.labels) * np.sum(self.labels == 1.)
        dr = max(dr, 1-dr)

        if normalize:
            self.normalize_data(norm_type)

    def leave_one_out_val(self):
        x_dists = [
        self.predict(self.data[i].reshape(1,-1), np.vstack((self.data[:i], self.data[i+1:])), i)
            for i in range(len(self.data))
        ]

        pred = np.array(x_dists).reshape(-1)

        return np.sum(pred == self.labels) / len(self.labels)

    def normalize_data(self, norm_type):
        if norm_type == "z_norm":
            self.data = self.data - self.data.mean(axis=0, keepdims =True)
            self.data /=  (self.data.std(axis=0, keepdims =True) + 1e-8)
        else: # default norm min,max
            self.data = self.data - self.data.min(axis=0, keepdims =True)
            self.data /=  (self.data.max(axis=0, keepdims =True) + 1e-8)

    def dist(self, x_1, x_2):
        '''
            Given a feature 1xd compute distance
        '''
        return np.sqrt(np.sum((x_1-x_2)**2, axis=1))

    def predict(self, x_1, x_2, i):
        dist = self.dist(x_1, x_2)
        min_idx = np.argmin(dist)
        # if dist[min_idx] != 0:
        #     print("MIN dist: ", dist[min_idx])

        if min_idx < i:
            return self.labels[min_idx]

        return self.labels[min_idx + 1]
