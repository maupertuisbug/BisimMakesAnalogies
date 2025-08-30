import torchrl 
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data import LazyTensorStorage
import torch
from tensordict import TensorDict
from datasets.collect_d4rl import D4RLCollector
from replay_buffer.common_buffer import CommonBuffer


def test1():

    source_dict = {'random' : torch.randn(3, 4), 
                    'zeroes' : torch.zeros(3, 4, 5)}

    batch_size= [3]
    td = TensorDict(source_dict, batch_size)
    print(td.shape)

    source_dict = {'random' : torch.randn(7, 4), 
                    'zeroes' : torch.zeros(7, 4, 5)}

    batch_size= [7]
    td1 = TensorDict(source_dict, batch_size)
    print(td1.shape)

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size = 1000))
    rb.extend(td)
    rb.extend(td1)
    print(len(rb))




if __name__ == "__main__":
    collector = D4RLCollector(env_id = 'AntMaze_Medium-v4')
    dataset = collector.collect_data()
    replay_buffer = MinariExperienceReplay(dataset_id = "antmaze/sbsac-v0", batch_size=32, root = '/root/.minari/datasets/')
    data = {'d4rl' : replay_buffer}
    cb = CommonBuffer(1000, 32, 67, 'dr4l', data)
    td = cb.sample()
    print("rd collected")
    