
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, p, n, phi):
        self.p = p
        self.n = n
        self.phi = phi

    def generate_data(self, standard):
        self.mu = np.zeros(self.p)
        if standard == True:
            self.Sigma = np.diag(np.ones(self.p))
        else:
            Q = np.random.random((self.p, self.p))
            self.Sigma = np.dot(np.dot(Q.T, np.diag(abs(np.random.normal(0, 1, self.p)))), Q)
        self.X = np.random.multivariate_normal(self.mu, self.Sigma, self.n)
        fX = stats.multivariate_normal.pdf(self.X, self.mu, self.Sigma)
        anomaly_threshold = np.percentile(fX, self.phi*100)
        self.Y = fX < anomaly_threshold

    def add_points_random(self, remove_idx):
        new_X = np.random.multivariate_normal(self.mu, self.Sigma, 1)
        self.X[remove_idx, :] = new_X

    def add_points_attack(self, remove_idx):
        new_X = np.array([4, 4])
        self.X[remove_idx, :] = new_X

    def add_points_shift_mean(self, remove_idx, t):
        self.mu = np.array([0, 2 + 0.001*t])
        new_X = np.random.multivariate_normal(self.mu, self.Sigma, 1)
        self.X[remove_idx, :] = new_X

data = Dataset(p=2, n=3000, phi=0.05)
data.generate_data(standard=True)

remove_pt = 55
for i in range(3):
    data.add_points_random(remove_pt)
    plt.figure()
    plt.scatter(data.X[data.Y == True, 0], data.X[data.Y == True, 1], color='r', s=1)
    plt.scatter(data.X[data.Y == False, 0], data.X[data.Y == False, 1], color='b', s=1)
    plt.scatter(data.X[remove_pt, 0], data.X[remove_pt, 1], color='k', s=20)
    plt.show()

data.add_points_attack(remove_pt)
plt.figure()
plt.scatter(data.X[data.Y == True, 0], data.X[data.Y == True, 1], color='r', s=1)
plt.scatter(data.X[data.Y == False, 0], data.X[data.Y == False, 1], color='b', s=1)
plt.scatter(data.X[remove_pt, 0], data.X[remove_pt, 1], color='k', s=20)
plt.show()


for i in range(5000):
    remove_pt = np.random.randint(0, data.n)
    data.add_points_shift_mean(remove_pt, i)
    if (i+1) % 1000 == 0:
        plt.figure()
        plt.scatter(data.X[:, 0], data.X[:, 1], color='k', s=1)
        plt.show()

# plt.figure()
# plt.scatter(data.X[data.Y==True, 0], data.X[data.Y==True, 1], color='r', s=1)
# plt.scatter(data.X[data.Y==False, 0], data.X[data.Y==False, 1], color='b', s=1)
# # plt.savefig(title + '.png')
# plt.show()
# plt.close
