import numpy as np
import pandas as pd
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt
from collections import namedtuple


# Source: https://github.com/llSourcell/AI_for_video_games_demo/blob/master/policy_iteration_demo.py

def run_episode(env, policy, gamma, render = True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        #if render:
        #env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(env, v, gamma = 1.0):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    v = np.zeros(env.nS)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma):
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iterations = 10
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            k=i+1
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy,k

def value_iteration(env, gamma):
    v = np.zeros(env.nS)  
    max_iterations = 1000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        #print(np.sum(np.fabs(prev_v - v)))
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k=i+1
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v,k
'''def plots(gamma, array, ylabel, title):
    plt.plot(gamma, array, color='b')
    plt.xticks(gamma)
    #plt.yticks(np.unique(array))
    plt.xlabel('Gamma')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()'''

EpisodeStats = namedtuple("Stats",["iters", "rewards"])
    
def plot_rewards(rewards, title, ylabel):
    plt.plot(rewards, color='r')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Episodes')
    plt.grid()
    plt.show()
    
def plot_iters(iters, title, ylabel):
    plt.plot(iters, color='r')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Episodes')
    plt.grid()
    plt.show()