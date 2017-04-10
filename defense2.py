import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from generate_data import Dataset

class BetterThanFrank:
    def __init__(self, buffer):
        self.buffer = buffer

    def indices(self, list, filtr=lambda x: bool(x)):
        return [i for i, x in enumerate(list) if filtr(x)]

    def GEM(self, NN=10, num_suspect=1, threshold=None):
        dist_mat = euclidean_distances(self.buffer, self.buffer)
        sorted_dist_mat = dist_mat
        sorted_dist_mat.sort(axis=1)
        avg_kNN_dist = np.mean(sorted_dist_mat[:, range(1, NN+1)], axis=1) # first neighbor is itself
        if threshold == None:
            idx_suspect_pts = avg_kNN_dist.argsort()[-num_suspect:]  # last points with highest distances
        else:
            idx_suspect_pts = self.indices(avg_kNN_dist, lambda x: x > threshold)
        return idx_suspect_pts

    def iGEM(self, NN=10, num_suspect=1, threshold=None):
        dist_mat = euclidean_distances(self.buffer, self.buffer)
        sorted_dist_mat = dist_mat
        sorted_dist_mat.sort(axis=1)
        avg_kNN_dist = np.mean(sorted_dist_mat[:, range(1, NN+1)], axis=1) # first neighbor is itself
        if threshold == None:
            idx_suspect_pts = avg_kNN_dist.argsort()[0:num_suspect]  # last points with highest distances
        else:
            idx_suspect_pts = self.indices(avg_kNN_dist, lambda x: x < threshold)
        return idx_suspect_pts

def main():
    data = Dataset(p=2, n=3000, phi=0.05)
    data.generate_data(standard=True, df=10)
    
    normal_data = data.generate_new_points(75)[1] # 75% of new points are normal
    suspect_data = np.random.uniform(2, 6, 50).reshape((25, 2)) # last 25 points are suspect
    buff_data = np.append(normal_data, suspect_data, axis=0)
    s_pts = BetterThanFrank(buff_data).iGEM(NN=30, num_suspect=25)
    
    plt.figure()
    plt.scatter(buff_data[-25:, 0], buff_data[-25:, 1], color='r', s=3)
    plt.scatter(buff_data[0:75, 0], buff_data[0:75, 1], color='k', s=3)
    for i in range(len(s_pts)):
        plt.scatter(buff_data[s_pts[i], 0], buff_data[s_pts[i], 1], color='m', s=5)
    plt.show()
    plt.savefig('defense.png')
    plt.close()

if __name__ == '__main__':
    main()
