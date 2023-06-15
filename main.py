
import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True
from env.create_Adacket_env import KernelSearch
from lib.agent import DDPG
from datetime import *
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Adacket')
    parser.add_argument('--job', default='train', type=str, help='support option: train/test')
    # env

    parser.add_argument('--dataset', default='ArticularyWordRecognition', type=str, help='dataset to use (mtsc/utsc)')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0, type=float, help='minimum dialation ratio')
    parser.add_argument('--rbound', default=1, type=float, help='maximum dialation ratio')
    parser.add_argument('--lcbound', default=0, type=float, help='minimum channel ratio')
    parser.add_argument('--rcbound', default=1, type=float, help='maximum channel ratio')
    parser.add_argument('--lobound', default=0, type=float, help='minimum channel out')
    parser.add_argument('--robound', default=1, type=float, help='maximum channel out')
    parser.add_argument('--reward', default='muti_reward', type=str, help='Setting the reward')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-2, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=5, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1, type=float, help='')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--rmsize', default=10, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=30, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=16, type=int, help='number of data batch size, default as sum samples')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default=save_path+'logs', type=str, help='')
    return parser.parse_args()
def env_test_model(env, args):
    return env.test_val_model()

def train(num_episode, agent, env, output):
    agent.is_training = True
    step = episode = episode_steps = 0
    step_ = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode

        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup: 
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)
        last = False
        startk  = time.time()
        observation2, reward, done, info, infobest = env.step(action, step_, observation,last,episode)
        observation2 = deepcopy(observation2)
        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])
        if episode % int(num_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        step_ +=1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)
        if done:  # end of episode
            step_ = 0    
            final_reward = T[-1][0]
            print("Updata net!")
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(r_t, s_t, s_t1, a_t, done) 
                if episode > args.warmup :  
                    agent.update_policy()
                    
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

    return infobest

def env_test(env, agent):
    env.test_val()


if __name__ == "__main__":
    save_path = './results/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    args = parse_args()  
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    args.data_root="./dataset/"
    now = datetime.now().strftime('%Y-%m-%d-%H.%M')
    env = KernelSearch(args.dataset,
                            preserve_ratio=1,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export')
    if args.job == 'train':
        nb_states =75 # 15*5
        nb_actions = 4 
        agent = DDPG(nb_states, nb_actions, args,env.channels)
        info = train(args.train_episode, agent, env, args.output)
        train_acc,val_acc, search_acc = env_test_model(env, args)
        print('Dataset:{} Train_acc: {:.4f}, Test_acc: {:.4f}, Parameters (M): {:.4f}, Memory (M): {:.4f}'.format(args.dataset, train_acc,val_acc, info['best_size']* 1e-6, info['feature']*1e-6)
                                                                        )
    elif args.job == 'test':
        env_test(env, args)
