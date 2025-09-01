# collect some dataset 
# sample 
# sample goals 
import os
# os.system('rm -rf /root/.minari/datasets/')
os.environ["MUJOCO_GL"] = "egl"
import numpy
import torch
from torchrl.envs.libs.gym import GymEnv
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import DoubleToFloat
import minari 
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from stable_baselines3 import SAC

# import for D4RL datasets
import gymnasium
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from gymnasium.envs.registration import register
from minari import DataCollector
gymnasium.register_envs(gymnasium_robotics)
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyTensorStorage
from tensordict.tensordict import TensorDict



class D4RLCollector:
    def __init__(self, env_id, type):
        self.env_id = env_id 
        env = gymnasium.make(self.env_id, render_mode="rgb_array")
        self.env = env
        
    
    def collect_minari_data(self):
        self.env = DataCollector(self.env)
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
            # take action using the agent
            obs, reward, done, trucated, info = self.env_agent.step(action) 
            rewards.append(reward)
            if done or trucated:
                print(numpy.mean(rewards))
                obs, info = self.env_agent.reset()
                _, _  = self.env.reset()
                ep+=1
                rewards = []

        print('Creating dataset for :', str(self.env_id + "/sbsac-v0"))
        self.dataset = self.env.create_dataset(
            dataset_id = self.env_id + "/sbsac-v0",
        )
        return self.dataset

    def collect_data(self):

        self.dataset = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size = 100000))
         # Env for the Agent
        self.env_agent = FlattenObservation(self.env)
        agent = SAC("MlpPolicy", self.env_agent)
        agent.learn(total_timesteps=100_000, log_interval=4)
        agent.save("sac")
        del agent
        self.agent = SAC.load("sac")
        ep = 0
        obs, info = self.env_agent.reset()
        obs_img = numpy.ascontiguousarray(self.env_agent.render())
        rewards = []

        while ep < 10:
            action, _states = self.agent.predict(obs, deterministic=True) 
            # take action using the agent
            next_obs, reward, done, trucated, info = self.env_agent.step(action) 
            next_obs_img = numpy.ascontiguousarray(self.env_agent.render())
            transition = TensorDict({
                            "obs" : torch.tensor(obs).to("cpu"),
                            "obs_img" : torch.tensor(obs_img).to("cpu"),
                            "action" : torch.tensor(action).to("cpu"),
                            "next_obs" : torch.tensor(next_obs).to("cpu"),
                            "next_obs_img" : torch.tensor(next_obs_img).to("cpu"),
                            "reward" : torch.tensor(reward, device="cpu"),
                            "done"  : torch.tensor(int(info['success']),device="cpu")
                        }, batch_size=[])
            self.dataset.add(transition)
            rewards.append(info['success'])
            
            if done or trucated:
                print(numpy.mean(rewards))
                obs, info = self.env_agent.reset()
                _, _  = self.env.reset()
                ep+=1
                rewards = []

        print('Creating dataset for :', str(self.env_id + "/sbsac-v0"))
        return self.dataset
        



