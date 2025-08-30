import torchrl 
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyTensorStorage
import torch
from tensordict import TensorDict



class CommonBuffer:
    def __init__(self, buffer_size, batch_size, random_key, type_id, **kwargs):
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.buffer = TensorDictReplayBuffer(storage = LazyTensorStorage(), batch_size = self.capacity).to("cpu")
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(random_key)
        self.build_offline_buffer(type_id, **kwargs)

    
    def build_offline_buffer(self, type_id, **kwargs):

        minari_buffer = kwargs.get(type_id, TensorDict({})) # this should return TensorDictType - agrees on the leading dimensions of all entries.

        if type_id == 'd4rl':
            self.buffer = minari_buffer

            

    def sample(self):
        return self.buffer.sample()
    
    def permute_sample(self, sample):

        random_index = torch.randperm(self.batch_size)
        return sample[random_index]

