import numpy as np
import pandas as pd
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt
from collections import namedtuple
np.random.seed(420)

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
            break
    return v

def policy_iteration(env, gamma):
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iterations = 1000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            iters=i+1
            print ('PI converged at: %d' %(i+1))
            break
        policy = new_policy
    return policy, iters

def value_iteration(env, gamma):
    v = np.zeros(env.nS)  
    max_iterations = 1000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            iters=i+1
            print ('VI converged at: %d' %(i+1))
            break
    return v,iters


def plots(gammas, array, yaxis, title):
    plt.plot(gammas, array, color='r')
    plt.xticks(gammas)
    plt.title(title)
    plt.ylabel(yaxis)
    plt.xlabel('Gammas')
    plt.grid()
    plt.savefig('img/'+title+'.png')
    plt.show()
    
def plot_rewards(rewards, title, ylabel):
    plt.plot(rewards, color='r')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Episodes')
    plt.grid()
    plt.savefig('img/'+title+'.png')
    plt.show()
    
def plot_iters(iters, title, ylabel):
    plt.plot(iters, color='r')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Episodes')
    plt.grid()
    plt.savefig('img/'+title+'.png')
    plt.show()