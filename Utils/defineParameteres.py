import argparse


def define_parameters():
    parser = argparse.ArgumentParser()
    # The parameters for the MA-DDPG
    parser.add_argument('--env_name', type=str, default='simple_adversary_v3', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=50,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=2e3, help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    parser.add_argument('--visdom', type=bool, default=True, help="Open the visdom")
    parser.add_argument('--size_win', type=int, default=1000, help="Open the visdom")
    # The parameters for the communication network
    parser.add_argument('--num_uav', type=int, default=3, help='The number of UAV')
    parser.add_argument('--num_mgu', type=int, default=30, help='The number of the MGU')
    parser.add_argument('--rang_max_mgu', type=float, default=200, help='The maximum radius for MGUs disruption')
    parser.add_argument('--max_vel', type=float, default=20, help='The maximum velocity of the MGUs')
    parser.add_argument('--delta', type=float, default=1, help='The length of the time slot')
    parser.add_argument('--high_uav_init', default=[50, 100, 150], help='The init height of the UAV')
    parser.add_argument('--obs_time', type=float, default=60, help='The observe time')
    parser.add_argument('--high_max', type=float, default=10, help='The maximum high of the UAVs in each time slot')
    parser.add_argument('--dis_max', type=float, default=60, help='The maximum high of the UAVs in each time slot')
    parser.add_argument('--beta_0', type=float, default=10 ** (-50 / 10), help='The channel gain when distance is 1m')
    parser.add_argument('--power_uav', type=float, default=10 ** ((30 - 30) / 10), help='The transmit power of the UAV')
    parser.add_argument('--noise', type=float, default=10 ** ((-220 - 30) / 10) * (10 ** 6),
                        help='The noise density of the UAV')
    args = parser.parse_args()
    return args
