import gym
import compiler_gym
import random
import os
from compiler_gym.envs import CompilerEnv
import math
import optuna

#stable baselines
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

f1data = []
f2data = []
f3data = []
f4data = []

def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def SaveEvalData(model, env, fileName, steps, oginstrcount, benchmark, obsspace):

    episodes = 100

    running_inst_cnt = 0
    best_inst_cnt = 100000000
    worst_inst_cnt = 0

    running_rewards = 0
    best_rewards = 100000000
    worst_rewards = 0

    for i in range (1, episodes + 1 ):
        obs = env.reset()
        
        done = False
        score = 0
        actions = []
        best_actions = []
        
        while not done:
            action, _ = model.predict(obs)
            actions.append(action)
            #print(action)
            obs, reward, done, info = env.step(action)
            #print(done)
            score += reward
        
        ##print(actions)
        instCnt = env.observation["IrInstructionCount"]
        running_inst_cnt += instCnt
        running_rewards += score

        if (score < worst_rewards):
            worst_rewards = score
        if (score > best_rewards):
            best_rewards = score

        if (instCnt > worst_inst_cnt):
            worst_inst_cnt = instCnt
        if (instCnt < best_inst_cnt):
            best_inst_cnt = instCnt
            best_actions = actions

        #full data sets
        #algo, step, instcount, reward, ratio
        f1data.append(fileName + ','+ str(steps) + ','+ str(env.observation["IrInstructionCount"]) + ','+ str(truncate(score,4)) + ','+ str(truncate(int(oginstrcount) / instCnt,2))+'\n')
   

    #inst data sets
    #algo, step, best_instcount, worst_count, avg_count
    f2data.append(fileName + ','+ str(steps) + ','+ str(best_inst_cnt) + ','+ str(worst_inst_cnt) + ','+str(truncate(running_inst_cnt / episodes,2))+'\n')

    #inst data sets
    #algo, step, best_instcount, worst_count, avg_count
    f3data.append(fileName + ','+ str(steps) +','+ str(truncate(score,4)) + ','+ str(truncate(score,4)) + ','+ str(truncate(running_rewards / episodes,2))+'\n')


class CompilerGymWrapperNormal(gym.Wrapper):

    def __init__(self, env: CompilerEnv):
        super(CompilerGymWrapperNormal, self).__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_count = 0

    def reset(self):
        self.action_count = 0
        return self.env.reset()

    def step(self, action):
        action = int(action)
        self.action_count += 1
        observation, reward, done, info = self.env.step(action)
        if (self.action_count > 40):
            done = True
        return observation, reward, done, info
    
def CreateModelAndLearn(net_arch_enabled, net_arch, algorithm, maxsteps, pbenchmark, observataionSpace, stepvalue):

    ##env = gym.make("llvm-autophase-ic-v0",benchmark="cbench-v1/bzip2",observation_space="Autophase")
    env = gym.make("llvm-autophase-ic-v0",benchmark=pbenchmark,observation_space=observataionSpace)

    wrapped_env_normal = CompilerGymWrapperNormal(env)

    env.reset()
    #Baseline Learning path
    log_path = os.path.join('Training_Comparisson_Algo','Logs_'+algorithm)

    #custom nn
    net_arch = [dict(pi=[128,128,128,128],vf=[128,128,128,128])]

    originalInstCount = str(env.observation["IrInstructionCount"])

    f1 = open(algorithm+"_details.txt", "a")
    f1.writelines(originalInstCount+'\n')
    f1.close()

    #define a model
    if (algorithm == 'PPO'):
        model = PPO('MlpPolicy', wrapped_env_normal, verbose=1)
    elif (algorithm == 'DQN'):
        model = DQN('MlpPolicy', wrapped_env_normal, verbose=1, policy_kwargs={"net_arch": [256, 256]})
    if (algorithm == 'A2C'):
        model = A2C('MlpPolicy', wrapped_env_normal, verbose=1)

    timestep_count = stepvalue

    while timestep_count <= maxsteps:
        
        print("Timestpe:"+str(timestep_count))
        model.learn(total_timesteps = stepvalue)

        if (timestep_count == maxsteps):
            Model_Path = os.path.join('Training','Saved_Models_PPO_Algo_Comp',algorithm+'_Model_algoComp_steps_'+str(timestep_count))
            model.save(Model_Path)

        SaveEvalData(model, wrapped_env_normal, algorithm, timestep_count,originalInstCount,pbenchmark,observataionSpace)
        timestep_count += stepvalue


def FinalFileWriteWithCreateMpdelAndLearn(net_arch_enabled, net_arch, algorithm, maxsteps, pbenchmark, observataionSpace, stepvalue):

    CreateModelAndLearn(net_arch_enabled, net_arch, algorithm, maxsteps,pbenchmark,observataionSpace,stepvalue)
    bench = pbenchmark.replace("/","_")


    f1_name = bench+'_'+algorithm+'_'+observataionSpace+"_fulldatapoints.csv"
    f2_name = bench+'_'+algorithm+'_'+observataionSpace+"_instr_count.csv"
    f3_name = bench+'_'+algorithm+'_'+observataionSpace+"_reward_details.csv"

    f1 = open(f1_name, "a")
    f2 = open(f2_name, "a")
    f3 = open(f3_name, "a")

    for i in f1data:
        f1.writelines(i)
    
    for i in f2data:
        f2.writelines(i)

    for i in f3data:
        f3.writelines(i)

    f1.close()
    f2.close()
    f3.close()

def Optimize(trial):
    env = gym.make("llvm-autophase-ic-v0",benchmark="cbench-v1/bzip2",observation_space="Autophase")
    wrapped_env_normal = CompilerGymWrapperNormal(env)

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)

    net_arch = [dict(pi=[256,256,256,256],vf=[256,256,256,256])]


    model = PPO('MlpPolicy', wrapped_env_normal,learning_rate=learning_rate, gae_lambda=gae_lambda, gamma=gamma, n_steps=n_steps, verbose=1, policy_kwargs={"net_arch": net_arch})
    model.learn(total_timesteps=10000)
    mean_reward, _ = evaluate_policy(model, wrapped_env_normal, n_eval_episodes=30)
    
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(Optimize, n_trials=20, n_jobs= 4)


print("Best HyperParameters: ", study.best_params)

##CreateModelAndLearn(False, 0, 'PPO', 60000,"cbench-v1/bzip2","Autophase",1000)
##FinalFileWriteWithCreateMpdelAndLearn(False, 0, 'DQN', 60000,"cbench-v1/bzip2","Autophase",1000)
##CreateModelAndLearn(False, 0, 'DQN', 60000,"cbench-v1/bzip2","Autophase",1000)