import os
import torch
from shutil import copyfile, copytree
import torch.nn as nn
import argparse
import givenData
import numpy as np
from gym.envs.registration import register


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 1:
            bias = self._bias.t().view(1, -1)
        elif x.dim() == 3:
            bias = self._bias.t().view(1, 1, -1)
        else:
            assert False

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def backup(timeStr, args, upper_policy=None):
    if args.evaluate:
        targetDir = os.path.join('./logs/evaluation', timeStr)
    else:
        targetDir = os.path.join('./logs/experiment', timeStr)

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    copyfile('attention_model.py', os.path.join(targetDir, 'attention_model.py'))
    copyfile('arguments.py', os.path.join(targetDir, 'arguments.py'))
    copyfile('envs.py', os.path.join(targetDir, 'envs.py'))
    copyfile('evaluation.py', os.path.join(targetDir, 'evaluation.py'))
    copyfile('evaluation_tools.py', os.path.join(targetDir, 'evaluation_tools.py'))
    copyfile('givenData.py', os.path.join(targetDir, 'givenData.py'))
    copyfile('graph_encoder.py', os.path.join(targetDir, 'graph_encoder.py'))
    copyfile('main.py', os.path.join(targetDir, 'main.py'))
    copyfile('rlagent.py', os.path.join(targetDir, 'rlagent.py'))
    copyfile('memory.py', os.path.join(targetDir, 'memory.py'))
    copyfile('tools.py', os.path.join(targetDir, 'tools.py'))
    copyfile('trainer.py', os.path.join(targetDir, 'trainer.py'))

    gymPath = './packing_envs'
    envName = args.id.split('-v')
    envName = envName[0]
    envPath = os.path.join(gymPath, envName)
    copytree(envPath, os.path.join(targetDir, envName))

    if upper_policy is not None:
        torch.save(upper_policy.state_dict(),
                   os.path.join(args.model_save_path, timeStr, 'upper-first-' + timeStr + ".pt"))


# Parsing PCT node from state returned in environment
def get_leaf_nodes(observation, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((observation.shape[0], -1, 9))
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes


def get_leaf_nodes_with_factor(observation, batch_size, internal_node_holder, leaf_node_holder):
    unify_obs = observation.reshape((batch_size, -1, 9))
    # unify_obs[:, :, 0:6] *= factor
    leaf_nodes = unify_obs[:, internal_node_holder:internal_node_holder + leaf_node_holder, :]
    return unify_obs, leaf_nodes


'''
Parsing the raw state returned in environment:

internal_nodes    : A packed item vector, [x1, y1, z1, x2, y2, z2, density(optional) ]
                    x1, y1, z1 are coordinates of a packed item
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are sizes of a packed item (a little different from the original paper,
                    these two description have similar performance.).
leaf_nodes        : A placement vector, [x1, y1, z1, x2, y2, z2]
                    x1, y1, z1 are coordinates of a placement.
                    x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                    x, y, z are  sizes of the current item after an axis-aligned orientation (a little different from the original paper,
                    these two description have similar performance.).
next_item         : The next item to be packed [density(optional), 0, 0,x, y, z]
                    x, y, z are  sizes of the current item.
invalid_leaf_nodes: The mask which indicates whether this placement is feasible.
full_mask         : The mask which indicates whether this node should be encode by GAT.
'''


def observation_decode_leaf_node(observation, internal_node_holder, internal_node_length, leaf_node_holder):
    internal_nodes = observation[:, 0:internal_node_holder, 0:internal_node_length]
    leaf_nodes = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 0:8]
    current_box = observation[:, internal_node_holder + leaf_node_holder:, 0:6]
    valid_flag = observation[:, internal_node_holder: internal_node_holder + leaf_node_holder, 8]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask


def decode_global_observation(observation, arg):
    """
    Decode the global observation of the order and bin agents
    :param observation: The observation returned by the environment, shape = (batch_size, (bin_num * internal_node_holder + buffer_size) * 9)
    :param arg: The arguments
    :return: The internal nodes, buffered boxes, full mask
    internal_nodes    : A packed item vector, [x1, y1, z1, x2, y2, z2, density(optional) ]
                        x1, y1, z1 are coordinates of a packed item
                        x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                        x, y, z are sizes of a packed item (a little different from the original paper,
                        these two description have similar performance.).
    buffered_boxes    : The buffered boxes to be packed [density(optional), 0, 0,x, y, z]
                        x, y, z are  sizes of the current item.
    full_mask         : The mask which indicates whether this node should be encoded by GAT.
                        The last bit of all nodes is the full mask.
    """

    observation = observation.reshape((observation.shape[0], -1, 9))
    internal_nodes = observation[:, 0:arg.bin_num * arg.internal_node_holder, 0:arg.internal_node_length]
    buffered_boxes = observation[:, arg.bin_num * arg.internal_node_holder:, 0:6]
    full_mask = observation[:, :, -1]
    return internal_nodes, buffered_boxes, full_mask


def decode_local_observation(observation, arg):
    """
    Decode the local observation of the location agent
    :param observation: The observation returned by the environment, shape = (batch_size, (internal_node_holder + leaf_node_holder + 1) * 9)
    :param arg: The arguments
    :return: The internal nodes, leaf nodes, current box, valid flag, full mask
    internal_nodes    : A packed item vector, [x1, y1, z1, x2, y2, z2, density(optional) ]
                        x1, y1, z1 are coordinates of a packed item
                        x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                        x, y, z are sizes of a packed item (a little different from the original paper,
                        these two description have similar performance.).
    leaf_nodes        : A placement vector, [x1, y1, z1, x2, y2, z2]
                        x1, y1, z1 are coordinates of a placement.
                        x2 = x1 + x, y2 = y1 + y, z2 = z1 + z
                        x, y, z are  sizes of the current item after an axis-aligned orientation (a little different from the original paper,
                        these two description have similar performance.).
    current_box       : The next item to be packed [density(optional), 0, 0,x, y, z]
                        x, y, z are  sizes of the current item.
    valid_flag        : The mask which indicates whether this placement is feasible.
                        The last bit of leaf_nodes is the valid flag.
    full_mask         : The mask which indicates whether this node should be encoded by GAT.
                        The last bit of all nodes is the full mask.
    """

    observation = observation.reshape((observation.shape[0], -1, 9))
    internal_nodes = observation[:, 0:arg.internal_node_holder, 0:arg.internal_node_length]
    leaf_nodes = observation[:, arg.internal_node_holder:arg.internal_node_holder + arg.leaf_node_holder, 0:8]
    current_box = observation[:, arg.internal_node_holder + arg.leaf_node_holder:, 0:6]
    valid_flag = observation[:, arg.internal_node_holder: arg.internal_node_holder + arg.leaf_node_holder, 8]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask


def load_policy(load_path, upper_policy):
    print(load_path)
    assert os.path.exists(load_path), 'File does not exist'
    pretrained_state_dict = torch.load(load_path, map_location='cpu')
    if len(pretrained_state_dict) == 2:
        pretrained_state_dict, ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if 'actor.embedder.layers' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v

    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    for k, v in load_dict.items():
        if len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    upper_policy.load_state_dict(load_dict, strict=True)
    print('Loading pre-train upper model', load_path)
    return upper_policy


def registration_envs():
    register(
        id='PackingDiscrete-v0',  # Format should be xxx-v0, xxx-v1
        entry_point='packing_envs.PackingDiscrete:PackingDiscrete',  # Expalined in envs/__init__.py
    )
    register(
        id='PackingContinuous-v0',
        entry_point='packing_envs.PackingContinuous:PackingContinuous',
    )
