import random
from typing import Iterator
import torch
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class RandomFrameSampler(Sampler):
    def __init__(self, data_infos, num_frames_per_file):
        self.data_infos = data_infos
        self.num_frames_per_file = num_frames_per_file
        self.sampled_data = self._sample_data()

    def __len__(self):
        return len(self.sampled_data)
    
    
    def _sample_data(self):
        file_name_to_frame_numbers = defaultdict(list)
        for i,data in enumerate(self.data_infos.data_infos):
            file_name = data[1]
            file_name_to_frame_numbers[file_name].append(i) #essentially storing the index

            
        sampled_data = []
        for file_name, data_list in file_name_to_frame_numbers.items():
            if len(data_list) <= self.num_frames_per_file:
                sampled_data.extend(data_list)
            else:
                sampled_data.extend(random.sample(data_list, self.num_frames_per_file))
        random.shuffle(sampled_data)
        return sampled_data


    def __iter__(self):
        return iter(self.sampled_data)