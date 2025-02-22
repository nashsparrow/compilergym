import gym
import compiler_gym
import random
import os
import numpy as np
from compiler_gym.envs import CompilerEnv
from gym import spaces

#stable baselines
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


##env = gym.make("llvm-autophase-ic-v0",benchmark="cbench-v1/bzip2",observation_space="Autophase",reward_space="IrInstructionCount")
env = gym.make("llvm-autophase-ic-v0",benchmark="cbench-v1/bzip2",observation_space="Autophase")


#subsetting the actions
actions = np.arange(1, 124, 1).tolist()
picked_actions = []
subset_actions = [[],[]]

i = 0
while i < 123:
    action = random.choice(actions)
    actions.remove(action)
    if (i % 2 ==0):
        subset_actions[0].append(action)
    else:
        subset_actions[1].append(action)
    i += 1

print(subset_actions)

#compilergym wrapper for baseline3
class CompilerGymWrapperChildRL(gym.Wrapper):
    def __init__(self, env: CompilerEnv, subSetActionSpace):
        super(CompilerGymWrapperChildRL, self).__init__(env)
        self.observation_space = env.observation_space
        self.subsetSpace = subSetActionSpace
        self.action_space = spaces.Discrete(len(self.subsetSpace))
        self.action_count = 0

    def reset(self):
        self.action_count = 0
        return self.env.reset()

    def step(self, action):
        action = int(action)
        self.action_count += 1
        observation, reward, done, info = self.env.step(action)
        if (self.action_count > 5): ##max subset = 15
            done = True
        return observation, reward, done, info

    def action(self,action):
        return self.subsetSpace[action]

    def reverse_action(self,action):
        return self.subsetSpace[action]
    
#creating root env
childEnv1 = CompilerGymWrapperChildRL(env,subset_actions[0])
childEnv2 = CompilerGymWrapperChildRL(env,subset_actions[1])

#creating models and learn
#Baseline Learning path
log_path = os.path.join('Training','Logs')

#custom nn
net_arch = [dict(pi=[128,128,128,128],vf=[128,128,128,128])]

#define a mode

cModel1 = PPO('MlpPolicy', childEnv1, verbose=1)
cModel2 = PPO('MlpPolicy', childEnv2, verbose=1, policy_kwargs={'net_arch':net_arch})

cModel1.learn(total_timesteps=10000)