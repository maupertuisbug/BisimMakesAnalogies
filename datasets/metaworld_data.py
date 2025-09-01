import os 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"
from OpenGL import EGL
display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
initialized = EGL.eglInitialize(display, None, None)
print("EGL initialized:", bool(initialized))
import numpy 
import torch
from torchrl.envs.libs.gym import GymEnv
from tensordict.tensordict import TensorDict
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyTensorStorage
from torchrl.envs import DoubleToFloat 
import minari 
from torchrl.data import ReplayBuffer 
from torchrl.data.replay_buffers import LazyMemmapStorage
from stable_baselines3 import SAC 
# imports for MetaWorld Datasets
import metaworld
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy

from gymnasium.wrappers import FlattenObservation
from minari import DataCollector
import matplotlib.pyplot as plt


class MetaWorldCollector:
    def __init__(self, env_id):
        self.env_id = str(env_id)
        mt1 = metaworld.MT1(self.env_id)
        task = list(mt1.train_tasks)[0]
        env = mt1.train_classes[self.env_id]()
        env.set_task(task)
        env.render_mode = "rgb_array"
        self.env = env

    def collect_minari_data(self):
        self.env = DataCollector(self.env)
        # Env for the Agent
        self.env_agent = FlattenObservation(self.env)
        agent = SAC("MlpPolicy", self.env_agent)
        agent.learn(total_timesteps=100, log_interval=4)
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
            rewards.append(info['success'])
            if done or trucated:
                print(numpy.mean(rewards))
                obs, info = self.env_agent.reset()
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
        agent.learn(total_timesteps=100, log_interval=4)
        agent.save("sac")
        del agent
        self.agent = SAC.load("sac")
        ep = 0
        obs, info = self.env_agent.reset()
        obs_img = numpy.ascontiguousarray(self.env_agent.render())
        rewards = []
        _obs  = [] 
        _obs_img = [] 
        _action = [] 
        _next_obs = [] 
        _next_obs_img = [] 
        _reward = [] 
        _done = []

        while ep < 10:
            action, _states = self.agent.predict(obs, deterministic=True) 
            # take action using the agent
            next_obs, reward, done, trucated, info = self.env_agent.step(action) 
            next_obs_img = numpy.ascontiguousarray(self.env_agent.render())
            _obs.append(obs)
            _obs_img.append(obs_img)
            _action.append(action)
            _next_obs.append(next_obs)
            _next_obs_img.append(next_obs_img)
            _reward.append(reward)
            _done.append(int(info['success']))
            # transition = TensorDict({
            #                 "obs" : torch.tensor(obs).to("cpu"),
            #                 "obs_img" : torch.tensor(obs_img).to("cpu"),
            #                 "action" : torch.tensor(action).to("cpu"),
            #                 "next_obs" : torch.tensor(next_obs).to("cpu"),
            #                 "next_obs_img" : torch.tensor(next_obs_img).to("cpu"),
            #                 "reward" : torch.tensor(reward, device="cpu"),
            #                 "done"  : torch.tensor(int(info['success']),device="cpu")
            #             }, batch_size=[])
            # self.dataset.add(transition)
            rewards.append(info['success'])
            
            if done or trucated:
                episode = minari.EpisodeData(ep, 
                                             observations = numpy.array(_obs),
                                             actions = numpy.array(_action),
                                             rewards = numpy.array(_reward),
                                             terminations = numpy.array(_done),
                                             truncations = numpy.array(_done),
                                             infos = {'obs_img' :  numpy.array(_obs_img)})
                transition = TensorDict({
                    'episodes' : episode
                }, batch_size=[])
                self.dataset.add(transition)
                print(numpy.mean(rewards))
                obs, info = self.env_agent.reset()
                ep+=1
                rewards = []
                _obs  = [] 
                _obs_img = [] 
                _action = [] 
                _next_obs = [] 
                _next_obs_img = [] 
                _reward = [] 
                _done = []

        print('Creating dataset for :', str(self.env_id + "/sbsac-v0"))
        return self.dataset


    def collect_minari_data_using_expert(self): 
        obs, info = self.env.reset()
        policy = SawyerPickPlaceV3Policy()
        done = False 
        epl = 0
        ep = 0
        while ep < 5 :
            epl = 0
            ep+=1
            obs, info = self.env.reset()
            done = False 
            while not done :
                epl+=1
                a = policy.get_action(obs)
                obs, _, _, _, info = self.env.step(a)
                pixels = numpy.ascontiguousarray(self.env.render())
                plt.imshow(pixels)
                plt.savefig('test.png')
                done = int(info['success']) == 1

        print('Episode Length :', epl)
        print('Creating dataset for :', str(self.env_id + "/sbsac-v0"))
        self.dataset = self.env.create_dataset(
            dataset_id = self.env_id + "/sbsac-v0",
        )
        return self.dataset

    def collect_data_using_expert(self): 
        self.dataset = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size = 100000))
        policy = SawyerPickPlaceV3Policy()
        done = False 
        epl = 0
        ep = 0
        obs, info = self.env.reset()
        obs_img = numpy.ascontiguousarray(self.env.render())
        rewards = []

        while ep < 10:
            action = policy.get_action(obs) 
            # take action using the agent
            next_obs, reward, done, trucated, info = self.env.step(action) 
            next_obs_img = numpy.ascontiguousarray(self.env.render())
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
                obs, info = self.env.reset()
                ep+=1
                rewards = []

        print('Creating dataset for :', str(self.env_id + "/sbsac-v0"))
        return self.dataset