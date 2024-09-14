import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from Utils.defineParameteres import define_parameters


class MGU:
    def __init__(self, par):
        self.par = par
        self.tra_mgu = []
        self.tra_lrc = []
        self.init_pos_mgu, self.init_vel_mgu, self.init_pos_lrc, self.init_vel_lrc = self.init_mgu()
        self.now_pos_mgu = np.copy(self.init_pos_mgu)
        self.now_vel_mgu = np.copy(self.init_vel_mgu)
        self.now_pos_lrc = np.copy(self.init_pos_lrc)
        self.now_vel_lrc = np.copy(self.init_vel_lrc)
        self.pos_cluster_center = self.obtain_pos_of_cluster_center()

    def init_mgu(self):
        init_pos_mgu = np.zeros((self.par.num_mgu, 2))
        init_vel_mgu = np.zeros((self.par.num_mgu, 2))
        init_pos_lrc = np.zeros(2)
        for index_mgu in range(self.par.num_mgu):
            theta = np.random.uniform(0, 2 * np.pi)
            dis = np.random.uniform(0, self.par.rang_max_mgu)
            init_pos_mgu[index_mgu, 0] = init_pos_lrc[0] + dis * np.cos(theta)
            init_pos_mgu[index_mgu, 1] = init_pos_lrc[1] + dis * np.sin(theta)
            init_vel_mgu[index_mgu, 0] = np.random.uniform(0, self.par.max_vel)
            init_vel_mgu[index_mgu, 1] = np.random.uniform(0, self.par.max_vel)
        init_vel_lrc = np.random.uniform(0, self.par.max_vel, size=2)
        self.tra_mgu.append(init_pos_mgu)
        self.tra_lrc.append(init_pos_lrc)
        return init_pos_mgu, init_vel_mgu, init_pos_lrc, init_vel_lrc

    def update_pos_vel(self):
        self.now_pos_lrc += self.now_vel_lrc * self.par.delta
        for index_mgu in range(self.par.num_mgu):
            self.now_pos_mgu[index_mgu, 0] += (self.now_vel_mgu[index_mgu, 0] + self.now_vel_lrc[0]) * self.par.delta
            self.now_pos_mgu[index_mgu, 1] += (self.now_vel_mgu[index_mgu, 1] + self.now_vel_lrc[1]) * self.par.delta
            self.now_vel_mgu[index_mgu, 0] = np.random.random()
            self.now_vel_mgu[index_mgu, 1] = np.random.random()
        self.now_vel_lrc = np.random.uniform(0, self.par.max_vel, size=2)

    def store_trajectory(self):
        self.tra_mgu.append(self.now_pos_mgu.copy())
        self.tra_lrc.append(self.now_pos_lrc.copy())

    def plot_current_pos(self):
        plt.scatter(self.init_pos_mgu[:, 0], self.init_pos_mgu[:, 1])
        plt.scatter(self.init_pos_lrc[0], self.init_pos_lrc[1], color='red')
        plt.legend(["the position of the MGUs", "the position of the LRC"])
        plt.xlabel("X(m)")
        plt.ylabel("Y(m)")
        plt.show()

    def plot_trajectory_lrc(self):
        length = len(self.tra_lrc)
        for i in range(length):
            plt.scatter(self.tra_lrc[i][0], self.tra_lrc[i][1])
        plt.show()

    def plot_trajectory_mgu(self, index_mgu):
        if index_mgu == -1:
            tra = np.reshape(np.array(self.tra_mgu), (-1, self.par.num_mgu, 2))
            for id_mgu in range(self.par.num_mgu):
                x = tra[:, id_mgu, 0]
                y = tra[:, id_mgu, 1]
                plt.plot(x, y, color='red')
        else:
            x = np.reshape(self.tra_mgu[:][index_mgu][0], (-1, 1))
            y = np.reshape(self.tra_mgu[:][index_mgu][1], (-1, 1))
            plt.plot(x, y, color='red')
        plt.show()

    def obtain_pos_of_cluster_center(self):
        kmeans = KMeans(n_clusters=self.par.num_uav)
        kmeans.fit(self.init_pos_mgu)
        centers = kmeans.cluster_centers_
        return centers

    def rest_mgu(self):
        self.now_pos_mgu = np.copy(self.init_pos_mgu)
        self.now_vel_mgu = np.copy(self.init_vel_mgu)
        self.now_pos_lrc = np.copy(self.init_pos_lrc)
        self.now_vel_lrc = np.copy(self.init_vel_lrc)
        self.tra_mgu = []
        self.tra_lrc = []


if __name__ == '__main__':
    # 1. Define the parameters
    par = define_parameters()
    mgu = MGU(par)
    mgu.plot_current_pos()
    for i in range(100):
        mgu.update_pos_vel()
        mgu.store_trajectory()
    mgu.plot_trajectory_lrc()
    mgu.plot_trajectory_mgu(-1)
