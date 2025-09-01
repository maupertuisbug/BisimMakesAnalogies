import torchrl 
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from minari.dataset._storages.arrow_storage import ArrowStorage
from minari.dataset.minari_dataset import MinariDataset
from torchrl.data import LazyTensorStorage
import torch
from tensordict import TensorDict
from datasets.d4rl_data import D4RLCollector
from datasets.metaworld_data import MetaWorldCollector
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
    # collector = D4RLCollector(env_id = 'AntMaze_Medium-v4', type = 'd4rl')
    collector = MetaWorldCollector(env_id = 'pick-place-v3')
    # dataset = collector.collect_data_using_expert()
    dataset = collector.collect_data()
    # dataset = collector.collect_minari_data()
    # dataset = collector.collect_minari_data_using_expert()
    # mt1 = metaworld.MT1('pick-place-v3')
    # task = list(mt1.train_tasks)[0]
    # env = mt1.train_classes['pick-place-v3']()
    # env.set_task(task)
    # storage = ArrowStorage('/root/.minari/datasets/metaworld/sbsac-v0', env.observation_space, env.action_space)
    # dataset = MinariDataset(storage)
    replay_buffer = dataset.sample(3)
    episodes  = replay_buffer['episodes']
    print(replay_buffer[0].observations['observation'].shape)
   
   