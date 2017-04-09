
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from bullshit import multivariate_t

class Dataset:
    def __init__(self, p, n, phi):
        self.p = p
        self.n = n
        self.phi = phi

    def generate_data(self, standard, df=None):
        self.df = df
        self.mu = np.zeros(self.p)
        if standard == True:
            self.Sigma = np.diag(np.ones(self.p))
        else:
            Q = np.random.random((self.p, self.p))
            self.Sigma = np.dot(np.dot(Q.T, np.diag(abs(np.random.normal(0, 1, self.p)))), Q)
        if df == None:
            import scipy.stats as stats
            self.X = np.random.multivariate_normal(self.mu, self.Sigma, self.n)
            fX = stats.multivariate_normal.pdf(self.X, self.mu, self.Sigma)
        else:
            self.X = multivariate_t.rvs(self.mu, self.Sigma, self.df, self.n)
            fX = multivariate_t.pdf(self.X, self.mu, self.Sigma, self.df)
        self.anomaly_threshold = np.percentile(fX, self.phi*100)
        self.Y = fX < self.anomaly_threshold

    def add_points_random(self, remove_idx):
        new_X = np.random.multivariate_normal(self.mu, self.Sigma, 1)
        self.X[remove_idx, :] = new_X

    def add_points_attack(self, remove_idx):
        new_X = np.array([4, 4])
        self.X[remove_idx, :] = new_X

    def add_points_shift_mean(self, remove_idx, t):
        if t % 100 == 0:
            self.mu = np.array([0, 0.5]) + self.mu
        else:
            self.mu = self.mu
        if self.df == None:
            new_X = np.random.multivariate_normal(self.mu, self.Sigma, 1)
        else:
            new_X = multivariate_t.rvs(self.mu, self.Sigma, self.df, 1)
        self.X[remove_idx, :] = new_X

    def generate_new_points(self, num_pts):
        if self.df == None:
            new_X = np.random.multivariate_normal(self.mu, self.Sigma, num_pts)
            fX = stats.multivariate_normal.pdf(new_X, self.mu, self.Sigma)
        else:
            new_X = multivariate_t.rvs(self.mu, self.Sigma, self.df, num_pts)
            fX = multivariate_t.pdf(new_X, self.mu, self.Sigma, self.df)
        new_Y = fX < self.anomaly_threshold

        return((new_Y, new_X))

# data = Dataset(p=2, n=3000, phi=0.05)
# data.generate_data(standard=True, df=10)
# print(data.generate_new_points(1))

# remove_pt = 55
# for i in range(3):
#     data.add_points_random(remove_pt)
#     plt.figure()
#     plt.scatter(data.X[data.Y == True, 0], data.X[data.Y == True, 1], color='r', s=1)
#     plt.scatter(data.X[data.Y == False, 0], data.X[data.Y == False, 1], color='b', s=1)
#     plt.scatter(data.X[remove_pt, 0], data.X[remove_pt, 1], color='k', s=20)
#     plt.show()
#
# data.add_points_attack(remove_pt)
# plt.figure()
# plt.scatter(data.X[data.Y == True, 0], data.X[data.Y == True, 1], color='r', s=1)
# plt.scatter(data.X[data.Y == False, 0], data.X[data.Y == False, 1], color='b', s=1)
# plt.scatter(data.X[remove_pt, 0], data.X[remove_pt, 1], color='k', s=20)
# plt.show()
#

#for i in range(5000):
#    remove_pt = np.random.randint(0, data.n)
#    data.add_points_shift_mean(remove_pt, i)
#    if (i+1) % 1000 == 0:
#        plt.figure()
#        plt.scatter(data.X[:, 0], data.X[:, 1], color='k', s=1)
#        plt.show()

# plt.figure()
# plt.scatter(data.X[data.Y==True, 0], data.X[data.Y==True, 1], color='r', s=1)
# plt.scatter(data.X[data.Y==False, 0], data.X[data.Y==False, 1], color='b', s=1)
# plt.savefig('data.png')
# plt.show()
# plt.close
