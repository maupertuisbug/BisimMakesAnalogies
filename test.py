import gymnasium as gym
import metaworld

env = gym.make('Meta-World/MT1', env_name='reach-v3')

obs = env.reset()
a = env.action_space.sample()
next_obs, reward, terminate, truncate, info = env.step(a)