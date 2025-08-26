# collect some dataset 
# sample 
# sample goals 
import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
import d4rl
from torchrl.envs.libs.gym import GymEnv
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import DoubleToFloat
import minari 
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from stable_baselines3 import SAC

from minari import DataCollector

# dataset = minari.load_dataset('D4RL/antmaze/umaze-diverse-v1', download = True)
# env  = dataset.recover_environment()

# gym.register_envs(gymnasium_robotics)
# all_envs = list(gym.registry)
# for env_id in sorted(all_envs):
#     print(env_id)
dataset_id = 'AntMaze_Medium-v4'

env = DataCollector(gym.make(dataset_id, render_mode="rgb_array"))
agent = SAC()

dataset = env.create_dataset(
    dataset_id = dataset_id,
    algorithm_name = "SAC")

rb = ReplayBuffer(storage=LazyMemmapStorage(1000), batch_size=10)
d = rb.sample()