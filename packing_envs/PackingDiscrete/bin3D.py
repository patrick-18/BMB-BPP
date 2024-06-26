from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import torch
import random

class PackingDiscrete(gym.Env):
    def __init__(self,
                 args,
                 **kwags):

        self.box_buffer_vec = None
        self.next_den = None
        self.next_box = None
        self.args = args
        self.setting = args.setting
        self.bin_size = args.container_size
        self.item_set = args.item_size_set
        self.data_name = args.dataset_path
        self.load_test_data = args.load_dataset
        self.internal_node_holder = args.internal_node_holder
        self.leaf_node_holder = args.leaf_node_holder
        self.next_holder = args.buffer_size
        self.shuffle = args.shuffle

        self.size_minimum = np.min(np.array(self.item_set))
        
        if self.setting == 2: self.orientation = 6
        else: self.orientation = 2
        
        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)

        self.spaces = [Space(*self.bin_size, self.size_minimum, self.internal_node_holder) for _ in range(self.args.bin_num)]

        # Generator for train/test data
        if not self.load_test_data:
            assert self.item_set is not None
            self.box_creator = RandomBoxCreator(self.item_set)
            assert isinstance(self.box_creator, BoxCreator)
        if self.load_test_data:
            self.box_creator = LoadBoxCreator(self.data_name)

        self.test = self.load_test_data
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
        self.action_space = gym.spaces.Box(low=0, high=0, shape=(0,))
        self.global_state_dim = (self.internal_node_holder + self.leaf_node_holder + 1) * 9
        self.box_buffer = []
        self.next_box_vec = np.zeros((1, 9))
        self.selected_bin_idx = 0

        self.LNES = args.lnes  # Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    # Calculate space utilization inside a bin.
    def get_box_ratio(self):
        coming_box = self.next_box
        space = self.spaces[self.selected_bin_idx]
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
                    space.plain_size[0] * space.plain_size[1] * space.plain_size[2])

    def reset(self):
        self.box_creator.reset()
        self.packed = []
        for space in self.spaces:
            space.reset()
        self.box_creator.generate_box_size()
        cur_observation = self.cur_observation()
        return cur_observation

    # Count and return all PCT nodes.
    def cur_observation(self):
        boxes = []
        while len(self.box_buffer) < self.args.buffer_size:
            next_box = self.gen_next_box()
            if self.test:
                if self.setting == 3:
                    next_den = next_box[3]
                else:
                    next_den = 1
                next_box = [int(next_box[0]), int(next_box[1]), int(next_box[2])]
            else:
                if self.setting < 3:
                    next_den = 1
                else:
                    next_den = np.random.random()
                    while next_den == 0:
                        next_den = np.random.random()
            self.box_buffer.append({'box': next_box, 'den': next_den})

        for space in self.spaces:
            boxes.append(space.box_vec)

        self.box_buffer_vec = np.zeros((self.next_holder, 9))
        for i in range(self.next_holder):
            self.box_buffer_vec[i, 3:6] = sorted(list(self.box_buffer[i]['box']))
            self.box_buffer_vec[i, 0] = self.box_buffer[i]['den']
            self.box_buffer_vec[i, -1] = 1

        return np.reshape(np.concatenate((*boxes, self.box_buffer_vec)), (-1))

    # Generate the next item to be placed.
    def gen_next_box(self):
        return self.box_creator.preview(1)[0]

    # Detect potential leaf nodes and check their feasibility.
    def get_possible_position(self, binAction, orderAction):
        self.next_box = self.box_buffer[orderAction]['box']
        self.next_den = self.box_buffer[orderAction]['den']
        self.selected_bin_idx = binAction
        if   self.LNES == 'EMS':
            allPostion = self.space.EMSPoint(self.next_box,  self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box,  self.setting)
        elif self.LNES == 'EP':
            allPostion = self.space.ExtremePoint2D(self.next_box, self.setting)
        elif self.LNES == 'CP':
            allPostion = self.space.CornerPoint(self.next_box, self.setting)
        elif self.LNES == 'FC':
            allPostion = self.space.FullCoord(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        for position in allPostion:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs

            if self.spaces[binAction].drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                leaf_node_idx += 1

            if leaf_node_idx >= self.leaf_node_holder: break

        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        self.next_box_vec = np.zeros((1, 9))
        self.next_box_vec[0, 3:6] = sorted(list(self.next_box))
        self.next_box_vec[0, 0] = self.next_den
        self.next_box_vec[0, -1] = 1

        return np.reshape(np.concatenate((self.spaces[binAction].box_vec, leaf_node_vec, self.next_box_vec)), (-1))

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = int(leaf_node[3] - leaf_node[0])
        y = int(leaf_node[4] - leaf_node[1])
        z = list(self.next_box)
        z.remove(x)
        z.remove(y)
        z = z[0]
        action = (0, int(leaf_node[0]), int(leaf_node[1]))
        next_box = (x, y, int(z))
        return action, next_box

    def step(self, action):
        if len(action) != 3: action, next_box = self.LeafNode2Action(action)
        else: next_box = self.next_box

        idx = [action[1], action[2]]
        bin_index = 0
        rotation_flag = action[0]
        succeeded = self.spaces[self.selected_bin_idx].drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)

        if not succeeded:
            reward = 0.0
            done = True
            ratio = [space.get_ratio() for space in self.spaces]
            info = {'counter': len(self.space.boxes), 'ratio': sum(ratio) / len(ratio),
                    'reward': sum(ratio) * 10 / len(ratio), 'Valid': True}
            return self.cur_observation(), reward, done, info

        ################################################
        ############# cal leaf nodes here ##############
        ################################################
        packed_box = self.spaces[self.selected_bin_idx].boxes[-1]

        if  self.LNES == 'EMS':
            self.spaces[self.selected_bin_idx].GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                                       packed_box.lx + packed_box.x,
                                                       packed_box.ly + packed_box.y,
                                                       packed_box.lz + packed_box.z])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        box_ratio = self.get_box_ratio()
        self.box_creator.drop_box()  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list
        reward = box_ratio * 10

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['Valid'] = True
        return self.cur_observation(), reward, done, info

