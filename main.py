import sys
import torch.cuda
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from tools import *
from envs import make_vec_envs
import numpy as np
import random
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from tools import registration_envs
from arguments import get_args
import gym
from memory import ReplayMemory
from rlagent import Agent
import copy

def main(args):

    # The name of this experiment, related file backups and experiment tensorboard logs will
    # be saved to '.\logs\experiment' and '.\logs\runs'
    # custom = input('Please input the name of this experiment: ')
    custom = 'test'
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)
    
    print('Using device:', device)

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Backup all py files and create tensorboard logs
    log_writer_path = './logs/runs/{}'.format('BMB-BPP-' + timeStr)
    writer = SummaryWriter(log_dir=log_writer_path)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    # backup(timeStr, args, None)


    tmp_env = gym.make(args.id, args=args)
    mem_num = args.num_processes
    mem_capacity = int(args.memory_capacity / mem_num)

    loc_args = copy.deepcopy(args)
    loc_args.action_space = args.internal_node_holder + args.leaf_node_holder + 1
    loc_args.obs_len = tmp_env.observation_space.shape[0]   # TODO: obs_len should be changed
    loc_args.level = 'location'
    loc_agent = Agent(loc_args)
    loc_mem = [ReplayMemory(args=loc_args, capacity=mem_capacity, obs_len=loc_args.obs_len) for _ in range(mem_num)]

    bin_args = copy.deepcopy(args)
    bin_args.action_space = args.bin_num
    bin_args.obs_len = tmp_env.global_state_dim   # TODO: obs_len should be changed
    bin_args.embedding_dim = 16
    bin_args.level = 'bin'
    bin_agent = Agent(bin_args)
    bin_mem = [ReplayMemory(args=bin_args, capacity=mem_capacity, obs_len=bin_args.obs_len) for _ in range(mem_num)]

    order_args = copy.deepcopy(args)
    order_args.action_space = args.buffer_size
    order_args.obs_len = tmp_env.global_state_dim # TODO: obs_len should be changed
    order_args.embedding_dim = 16
    order_args.level = 'order'
    order_agent = Agent(order_args)
    order_mem = [ReplayMemory(args=order_args, capacity=mem_capacity, obs_len=order_args.obs_len) for _ in range(mem_num)]

    # Create parallel packing environments to collect training samples online
    envs = make_vec_envs(args, './logs/runinfo', True)

    # Perform all training.
    trainer = Trainer(writer, timeStr, [loc_agent, bin_agent, order_agent], [loc_mem, bin_mem, order_mem])
    trainer.train_q_value(envs, args)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    args.continuous = True

    # args.load_dataset = True
    # args.dataset_path = './dataset/setting2_continuous.pt'

    main(args)

