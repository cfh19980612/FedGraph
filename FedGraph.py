import numpy as np
import torch
import sys
import argparse
import time
import matplotlib.pyplot as plt
import utils
import os
import pandas as pd

from TD3 import TD3
from DDPG_N import DDPG
# from DDPG import Agent
from Environment import gcnEnv

client = 1
state_dim = client*client
action_dim = client*2
max_action = 25
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
file_name = ""

def Initialization():

    ######################################DDPG######################################
    # hyper-parameter for DDPG

    parser_ddpg = argparse.ArgumentParser(description = "Hyper-parameters for DRL")
    parser_ddpg.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser_ddpg.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser_ddpg.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser_ddpg.add_argument("--start_timesteps", default=32, type=int)# Time steps initial random policy is used
    parser_ddpg.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser_ddpg.add_argument("--max_timesteps", default=1, type=int)   # Max time steps to run environment
    parser_ddpg.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser_ddpg.add_argument("--batch_size", default=32, type=int)      # Batch size for both actor and critic
    parser_ddpg.add_argument("--discount", default=0.99)                 # Discount factor
    parser_ddpg.add_argument("--tau", default=0.005)                     # Target network update rate
    parser_ddpg.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser_ddpg.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser_ddpg.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser_ddpg.add_argument("--save_model", default = True)        # Save model and optimizer parameters
    parser_ddpg.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser_ddpg.add_argument("--dataset", type=str, default="cora",       # dataset
                    help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser_ddpg.add_argument("--gpu", type=int, default=-1,
                    help="gpu")
    args_ddpg = parser_ddpg.parse_args()
    kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount":args_ddpg.discount,
		"tau": args_ddpg.tau,
	}
    # kwargs["policy_noise"] = args_ddpg.policy_noise * max_action
    # kwargs["noise_clip"] = args_ddpg.noise_clip * max_action
    # kwargs["policy_freq"] = args_ddpg.policy_freq
    agent = DDPG(**kwargs)
    # agent = Agent(state_size=state_space, action_size=action_space, random_seed=10)  # agent

    # environment init
    env = gcnEnv(client, args_ddpg.dataset)
    ######################################GCN######################################
    file_name = f"{args_ddpg.policy}_{env.args.dataset}_{args_ddpg.seed}"

    return env, agent, args_ddpg

def Local_Training():

    return None

def Global_Agg():
    return None

def DRL_training(agent, env, args_ddpg):
    print(args_ddpg)
    if args_ddpg.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # DRL parameters
    times = 0
    print_every=100
    max_t = 10
    # scores_deque = deque(maxlen=print_every)
    scores = []
    max_value = 500
    t = 0
    # initial env and agent
    for episode in range(args_ddpg.max_timesteps):
        score = 0
        t0 = time.time()
        state = env.reset()
        start_time = time.time()
        for epoch in range(env.args.n_epochs):
            t += 1

            # choose an action
            action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * args_ddpg.expl_noise, size=action_dim)).clip(-max_action, max_action)

            # training information
            Train_info = (episode, epoch, t0)

            # take an action
            next_state, reward, acc, time_cost , done= env.step(action,Train_info)

            # store the experience and update the env
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state

            # store the total return
            score += reward

            if t >= args_ddpg.start_timesteps:
                agent.train(replay_buffer, args_ddpg.batch_size)

            if done or epoch == env.args.n_epochs - 1:
                break
            # if done:
            #     break
        end_time = time.time()
        print('Training time: ', end_time-start_time)

        scores.append(score)

        print()
        print('---------------------------------------------------------------------------------------------------')
        print('\rEpisode {}\tAverage Score: {:.2f}\tAccuracy: {}'.format(episode,scores[episode],acc),flush=True)
        print('---------------------------------------------------------------------------------------------------')
        print()
        # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Total Return')
    plt.xlabel('Episode')
    plt.savefig('./test2.jpg')
    plt.show()
    # save model
    if args_ddpg.save_model:
        agent.save(f"./models/{file_name}")

def DRL_test(agent, env):
    x_time = []
    y_accuracy = []
    state = env.reset()
    agent.load(f"./models/{file_name}")
    episode = 0
    print('sampling time: ', agent_time_end - agent_time_start)
    for epoch in range(env.args.n_epochs):
        # choose an action
        action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * args_ddpg.expl_noise, size=action_dim)).clip(-max_action, max_action)

        # training information
        Train_info = (episode, epoch, t0)

        # take an action
        next_state, reward, acc, time_cost, done= env.step(action,Train_info)

        # update the env
        state = next_state

        x_time.append(time_cost)
        y_accuracy.append(acc)
        if done or epoch == env.args.n_epochs - 1:
            dataframe = pd.DataFrame(x_time, columns=['X'])
            dataframe = pd.concat([dataframe, pd.DataFrame(y_accuracy,columns=['Y'])],axis=1)
            dataframe.to_csv(f"./result/{file_name}.csv",header = False,index=False,sep=',')
            break

def run():
    return None

if __name__ == '__main__':

    env, agent, args_ddpg = Initialization()
    DRL_training(agent, env, args_ddpg)
    DRL_test(agent, env)
