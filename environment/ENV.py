import numpy as np
from matplotlib import pyplot as plt
from networkx.algorithms.bipartite import color

from Utils.defineParameteres import define_parameters
from mobildGroundUsers.MGU import MGU


class ENV:
    def __init__(self, par, agent, mgu):
        self.par = par
        self.agent = agent
        self.mgu = mgu
        self.pos_uav_now = self.init_pos_uav()
        self.tra_uav = []
        self.agents = ['agent' + str(i) for i in range(self.par.num_uav)]

    def reset(self):
        obs_init = np.zeros((self.par.num_uav, self.par.dim_state))
        self.pos_uav_now = self.init_pos_uav()
        self.mgu.rest_mgu()
        self.tra_uav = []
        self.tra_uav.append(self.pos_uav_now.copy())
        # xu1 yu1 h1 xu2 yu2 h2 xu3 yu3 h3 x_lrc y_lrc d1 d2 d3
        for index_uav in range(self.par.num_uav):
            obs_init[index_uav, 0:3] = self.pos_uav_now[index_uav]
            obs_init[index_uav, 3:5] = self.mgu.init_pos_lrc
            obs_init[index_uav, 5] = self.calculate_distance_uav_to_lrc(index_uav)
            obs_init[index_uav, 6:8] = self.mgu.init_vel_lrc
        return obs_init

    def step(self, action):
        # 更新UAV位置
        for index_uav in range(self.par.num_uav):
            self.pos_uav_now[index_uav, 0] += action[index_uav, 1] * np.cos(action[index_uav, 0])
            self.pos_uav_now[index_uav, 1] += action[index_uav, 1] * np.sin(action[index_uav, 1])
            self.pos_uav_now[index_uav, 2] += action[index_uav, 2]
        self.tra_uav.append(self.pos_uav_now.copy())
        # 更新MGU位置
        self.mgu.update_pos_vel()
        self.mgu.store_trajectory()
        # 计算奖励
        dis_uav_to_mgu = self.calculate_distance_uav_to_mgu()
        gain = self.par.beta_0 / (dis_uav_to_mgu ** 2)
        sinr = self.par.power_uav * gain / self.par.noise

        rate = np.log2(1 + sinr)
        sum_rate = np.sum(rate, 1)
        reward_each_agent = sum_rate / 50000
        for index_uav in range(self.par.num_uav):
            reward_each_agent[index_uav] -= self.calculate_distance_uav_to_lrc(index_uav) / 10000
        reward_total = np.sum(reward_each_agent)
        # 计算下一状态
        obs_next = np.zeros((self.par.num_uav, self.par.dim_state))
        # xu1 yu1 h1 xu2 yu2 h2 xu3 yu3 h3 x_lrc y_lrc d1 d2 d3
        for index_uav in range(self.par.num_uav):
            obs_next[index_uav, 0:3] = self.pos_uav_now[index_uav]
            obs_next[index_uav, 3:5] = self.mgu.now_pos_lrc
            obs_next[index_uav, 5] = self.calculate_distance_uav_to_lrc(index_uav)
            obs_next[index_uav, 6:8] = self.mgu.now_vel_lrc
        done = False
        return obs_next, reward_each_agent, reward_total, done

    def init_pos_uav(self):
        pos_uav = np.zeros((self.par.num_uav, 3))
        for index_uav in range(self.par.num_uav):
            pos_uav[index_uav, 0:2] = self.mgu.pos_cluster_center[index_uav]
            pos_uav[index_uav, 2] = self.par.high_uav_init[index_uav]
        return pos_uav

    def calculate_distance_uav_to_lrc(self, index_uav):
        dis = np.sqrt(np.sum((self.pos_uav_now[index_uav, 0:2] - self.mgu.now_pos_lrc) ** 2) + self.pos_uav_now[
            index_uav, 2] ** 2)
        return dis

    def calculate_distance_uav_to_mgu(self):
        dis = np.zeros((self.par.num_uav, self.par.num_mgu))
        for index_uav in range(self.par.num_uav):
            dis[index_uav, :] = np.sqrt(
                np.sum((self.pos_uav_now[index_uav, 0:2] - self.mgu.now_pos_mgu) ** 2, 1) + self.pos_uav_now[
                    index_uav, 2] ** 2)
        return dis

    def plot_uav_trajectory(self):
        tra_uav = np.array(self.tra_uav)
        for index_uav in range(self.par.num_uav):
            x = tra_uav[:, index_uav, 0]
            y = tra_uav[:, index_uav, 1]
            plt.plot(x, y)
        plt.legend(["agent0", "agent1", "agent2"])
        plt.scatter(tra_uav[0, :, 0], tra_uav[0, :, 1])
        plt.show()

    def plot_state_environment_now(self):
        # 用户分布
        plt.scatter(self.mgu.now_pos_mgu[:, 0], self.mgu.now_pos_mgu[:, 1], color='blue')
        # 参考中心
        plt.scatter(self.mgu.now_pos_lrc[0], self.mgu.now_pos_lrc[1], color='red')
        # UAV
        plt.scatter(self.pos_uav_now[:, 0], self.pos_uav_now[:, 1], color='black')
        plt.show()
        # 参考中心


if __name__ == '__main__':
    # 1. Define the parameters
    par = define_parameters()
    mgu = MGU(par)
    agent = dict()
    for index_agent in range(par.num_uav):
        agent[index_agent] = MADDPGAGENT.AGENT(par)
    env = ENV(par, agent, mgu)
    state_now = env.reset()
    env.plot_state_environment_now()
    for index_step in range(int(par.obs_time / par.delta)):
        action = np.zeros((par.num_uav, par.dim_action))
        for index_agent in range(par.num_uav):
            action[index_agent, :] = agent[index_agent].choose_action(state_now[index_agent])
        state_next, reward_each_agent, reward_total, done = env.step(action)
    env.plot_state_environment_now()
    env.reset()
    env.plot_state_environment_now()
