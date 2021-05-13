import torch
import numpy as np
import random
import gym
import dqn
from config import CartPole, Pong
from train import train
from evaluate import evaluate_policy
import os

from utils import preprocess

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENV_CONFIGS = {
    'CartPole-v0': CartPole,
    'Pong-v0': Pong
}


def training(env):
    train(env)


def evaluating(env):
    path = f'/Users/haradys/Documents/Data_Science_Master/RL/Project/models/{env}_best_new.pt'
    n_eval_episodes = 10
    render = True
    save_video = False
    env_name = env

    # Initialize environment and config
    env_config = ENV_CONFIGS[env]
    env = gym.make(env)

    if save_video:
        env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force=True)

    # Load model from provided path.
    dqn = torch.load(path, map_location=torch.device('cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, env_name, n_eval_episodes, render=render and not save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {n_eval_episodes} episodes.')

    env.close()


if __name__ == "__main__":
    #env = 'CartPole-v0'
    print('Start')
    env = 'Pong-v0'
    training(env)
    print('End')
    #evaluating(env)

    '''
    network = dqn.DQN(Pong)
    env = gym.make('Pong-v0')
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                                          frame_skip=1, noop_max=30)
    obs = env.reset()
    n_obs = env.reset()
    print('shape of image', obs.shape)
    obs = preprocess(obs, env='Pong')
    n_obs = preprocess(n_obs, env="Pong")
    print('size of obs', obs.shape)
    obs = 4*[obs.unsqueeze(0)]
    n_obs = 4*[n_obs.unsqueeze(0)]
    obs_stack = torch.cat(obs)
    n_obs_stack = torch.cat(n_obs)
    print('before forward:', obs_stack.size())
    obs_stack = obs_stack.unsqueeze(0)
    n_obs_stack = n_obs_stack.unsqueeze(0)
    print(f'after unsqueezing: {obs_stack.size()}, {n_obs_stack.size()}')
    obs_stack = torch.cat((obs_stack, n_obs_stack), dim=0)
    print(f"new dim: {obs_stack.size()}")
    network.forward(obs_stack)
    #print(CartPole)
    network = dqn.DQN(CartPole)
    #print(network)
    b = torch.randint(low=0, high=10, size=(5,4)).float()
    ind = torch.randint(low=0, high=4, size=(5,1))
    #t = torch.tensor([[1,2,3,4], [11,2,7,0]]).type(torch.LongTensor)
    print(f'b:\n{b}')
    print(f'ind:\n{ind}')
    print(torch.gather(b, dim=1, index=ind))
    obs = network.forward(b)
    #print(obs)
    #print(torch.max(obs, dim=1))
    '''