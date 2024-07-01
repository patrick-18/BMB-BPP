import os
import numpy as np
import torch
from tools import decode_local_observation
from tqdm import trange
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

np.set_printoptions(threshold=np.inf)
import time


# Distributed training for online packing policy
def learningPara(T, priority_weight_increase, model_save_path, dqn, mem, timeStr, args, counter, lock, sub_time_str):
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr + '-loss')
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)
    targetCounter = T
    checkCounter = T
    logCounter = T
    timeStep = T
    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print('Distributed Training Start')
    torch.set_num_threads(1)
    while True:
        if not lock.value:
            for i in range(len(mem)):
                mem[i].priority_weight = min(mem[i].priority_weight + priority_weight_increase,
                                             1)  # Anneal importance sampling weight β to 1

            dqn.reset_noise()
            loss = dqn.learn(mem)  # Train with n-step distributional double-Q learning

            # Update target network
            if timeStep - targetCounter >= args.target_update:
                targetCounter = timeStep
                dqn.update_target_net()

            if timeStep % args.checkpoint_interval == 0:
                sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

            # Checkpoint the network #
            if (args.checkpoint_interval != 0) and (timeStep - checkCounter >= args.save_interval):
                checkCounter = timeStep
                dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

            if timeStep - logCounter >= args.print_log_interval:
                logCounter = timeStep
                writer.add_scalar("Training/Value loss", loss.mean().item(), logCounter)

            timeStep += 1
        else:
            time.sleep(0.5)


class Trainer(object):
    def __init__(self, writer, timeStr, dqns, mems):
        self.writer = writer
        self.timeStr = timeStr
        self.loc_dqn, self.bin_dqn, self.order_dqn = dqns
        self.loc_mem, self.bin_mem, self.order_mem = mems

    def train_q_value(self, envs, args):
        global counter, loc_loss, bin_loss, order_loss
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)

        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        global_state = envs.reset()

        batchX = torch.arange(args.num_processes)

        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        if args.distributed:
            counter = mp.Value('i', 0)
            lock = mp.Value('b', False)
        # Training loop
        self.loc_dqn.train()
        self.bin_dqn.train()
        self.order_dqn.train()
        for T in trange(1, args.T_max + 1):

            if T % args.replay_frequency == 0 and not args.distributed:
                self.loc_dqn.reset_noise()  # Draw a new set of noisy weights
                self.bin_dqn.reset_noise()
                self.order_dqn.reset_noise()

            binAction = self.bin_dqn.act(global_state, mask=None)
            orderAction = self.order_dqn.act(global_state, mask=None)
            local_state = envs.get_possible_position(binAction.cpu().numpy(), orderAction.cpu().numpy())
            local_state = torch.from_numpy(np.array(local_state)).float().to(args.device)

            # decode the observation
            # TODO: change the observation of single bin to the observation of multiple bins
            _, leaf_nodes, _, leaf_mask, _ = decode_local_observation(local_state, args)

            locAction = self.loc_dqn.act(local_state, mask=leaf_mask)  # Choose an action greedily (with noisy weights)
            selected_leaf_nodes = leaf_nodes[batchX, locAction.squeeze()]

            next_state, reward, done, infos = envs.step(selected_leaf_nodes.cpu().numpy())  # Step

            validSample = []
            for _ in range(len(infos)):
                validSample.append(infos[_]['Valid'])
                if done[_] and infos[_]['Valid']:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])

            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            for i in range(args.num_processes):
                if validSample[i]:
                    self.loc_mem[i].append(local_state[i], locAction[i], reward[i], done[i])  # Append transition to memory
                    self.bin_mem[i].append(global_state[i], binAction[i], reward[i], done[i])
                    self.order_mem[i].append(global_state[i], orderAction[i], reward[i], done[i])

            if args.distributed:
                counter.value = T
                if T == args.learn_start:
                    learningProcess = mp.Process(target=learningPara, args=(
                    T, priority_weight_increase, model_save_path, self.dqn, self.mem, self.timeStr, args, counter, lock,
                    sub_time_str))
                    learningProcess.start()
            else:
                # Train and test
                if T >= args.learn_start:
                    for i in range(args.num_processes):
                        self.loc_mem[i].priority_weight = min(self.loc_mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                        self.bin_mem[i].priority_weight = min(self.bin_mem[i].priority_weight + priority_weight_increase, 1)
                        self.order_mem[i].priority_weight = min(self.order_mem[i].priority_weight + priority_weight_increase, 1)

                    if T % args.replay_frequency == 0:
                        loc_loss = self.loc_dqn.learn(self.loc_mem)  # Train with n-step distributional double-Q learning
                        bin_loss = self.bin_dqn.learn(self.bin_mem)
                        order_loss = self.order_dqn.learn(self.order_mem)
                    # Update target network
                    if T % args.target_update == 0:
                        self.loc_dqn.update_target_net()
                        self.bin_dqn.update_target_net()
                        self.order_dqn.update_target_net()

                    # Checkpoint the network #
                    if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                        if T % args.checkpoint_interval == 0:
                            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                        self.loc_dqn.save(model_save_path, 'locAgent-checkpoint-{}.pt'.format(sub_time_str))
                        self.bin_dqn.save(model_save_path, 'binAgent-checkpoint-{}.pt'.format(sub_time_str))
                        self.order_dqn.save(model_save_path, 'orderAgent-checkpoint-{}.pt'.format(sub_time_str))

                    if T % args.print_log_interval == 0:
                        self.writer.add_scalar("Training/loc loss", loc_loss.mean().item(), T)
                        self.writer.add_scalar("Training/bin loss", bin_loss.mean().item(), T)
                        self.writer.add_scalar("Training/order loss", order_loss.mean().item(), T)
                        if len(episode_rewards) != 0:
                            self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                            self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                            self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
                        if len(episode_ratio) != 0:
                            self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
                        if len(episode_counter) != 0:
                            self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)

            if np.all(done):  # Terminal state
                global_state = envs.reset()
            else:
                global_state = next_state
