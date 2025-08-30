# collect some dataset 
# sample 
# sample goals 
import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium
import gymnasium_robotics
import numpy

gymnasium.register_envs(gymnasium_robotics)
import datasets.collect_d4rl as collect_d4rl
from torchrl.envs.libs.gym import GymEnv
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import DoubleToFloat
import minari 
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from stable_baselines3 import SAC
from gymnasium.wrappers import FlattenObservation

from minari import DataCollector

# dataset = minari.load_dataset('D4RL/antmaze/umaze-diverse-v1', download = True)
# env  = dataset.recover_environment()

# gym.register_envs(gymnasium_robotics)
# all_envs = list(gym.registry)
# for env_id in sorted(all_envs):
#     print(env_id)
# dataset_id = 'AntMaze_Medium-v4'

# env = DataCollector(gym.make(dataset_id, render_mode="rgb_array"))
# agent = SAC()

# dataset = env.create_dataset(
#     dataset_id = dataset_id,
#     algorithm_name = "SAC")

# rb = ReplayBuffer(storage=LazyMemmapStorage(1000), batch_size=10)
# d = rb.sample()


class D4RLCollector:
    def __init__(self, env_id):
        self.env_id = env_id 
    
    def collect_data(self):
        self.env = DataCollector(gymnasium.make(self.env_id, render_mode="rgb_array")) # return a dict of observation and goals
        
        # Env for the Agent
        self.env_agent = FlattenObservation(self.env)
        agent = SAC("MlpPolicy", self.env_agent)
        agent.learn(total_timesteps=10, log_interval=4)
        agent.save("sac")
        del agent

        self.agent = SAC.load("sac")
        ep = 0
        obs, info = self.env_agent.reset()
        rewards = []
        
        while ep < 10:
            action, _states = self.agent.predict(obs, deterministic=True)

            # take action in the datacollector env for storage
            # obs_d, reward_d, done_d, trucated_d, info_d = self.env.step(action) 

            # take action using the agent
            obs, reward, done, trucated, info = self.env_agent.step(action) 
            rewards.append(reward)
            if done or trucated:
                print(numpy.mean(rewards))
                obs, info = self.env_agent.reset()
                _, _  = self.env.reset()
                ep+=1
                rewards = []

        self.dataset = self.env.create_dataset(
            dataset_id = "antmaze/sbsac-v0",
        )
        return self.dataset
        



