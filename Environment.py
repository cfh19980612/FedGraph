import time 
import math
import gym
import numpy as np
import warnings

# from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from collections import deque
from DDPG import Agent
from GCN_processor import *
from tkinter import _flatten

warnings.filterwarnings('ignore')

dataset = 'Pubmed'

# define the environment
class gcnEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, client, dataset):
        ######################################GCN######################################
        # hyper-parameter for GCN
        parser_gcn = argparse.ArgumentParser(description = "Hyper-parameters for GCN")
        parser_gcn.add_argument("--dropout", type=float, default=0.5,
                            help="dropout probability")
        parser_gcn.add_argument("--gpu", type=int, default=-1,
                            help="gpu")
        parser_gcn.add_argument("--dataset", type=str, default="cora",       # dataset
                            help="Dataset name ('cora', 'citeseer', 'pubmed').")
        parser_gcn.add_argument("--lr", type=float, default=1e-2,
                            help="learning rate")
        parser_gcn.add_argument("--n-epochs", type=int, default=200,
                            help="number of training epochs")
        parser_gcn.add_argument("--n-hidden", type=int, default=16,
                            help="number of hidden gcn units")
        parser_gcn.add_argument("--n-layers", type=int, default=1,
                            help="number of hidden gcn layers")
        parser_gcn.add_argument("--weight-decay", type=float, default=5e-4,
                            help="Weight for L2 loss")
        parser_gcn.add_argument("--self-loop", action='store_true',
                            help="graph self-loop (default=True)")
        parser_gcn.add_argument("--batch-size", type=int, default=256,
                            help="batch size")
        parser_gcn.add_argument("--client", type=int, default=client,
                                help="Clients")
        parser_gcn.set_defaults(self_loop=True)
        args_gcn = parser_gcn.parse_args()
        ######################################GCN######################################
        self.dataset = dataset
        self.args = args_gcn
        print(self.args)
        # GCN
        self.client = self.args.client
        self.global_model = None
        self.global_GCN_info = None
        self.Model = [None for i in range (self.client)]
        self.GCN_info = [None for i in range (self.client)]
        self.Graph = [None for i in range (self.client)]
        self.dur = []

        # target
        if self.dataset == 'cora': self.target = 0.89
        elif self.dataset == 'citeseer': self.target = 0.72
        elif self.dataset == 'pubmed': self.target = 0.8
        elif self.dataset == 'reddit': self.target = 0.99

        # pca
        self.pca = PCA(n_components = self.client)
        # gpu?
        if self.args.gpu < 0:
            self.cuda = False
        else:
            self.cuda = True


    def step(self, action, Train_info):
        Acc = 0
        Time_cost = [0 for i in range (self.client)]
        dur = [None, None]
        # Acc = [0 for i in range (self.client)]
        action = action.tolist()
        action = list(_flatten(action))
        # sampling
        Sampling = [np.array([]) for i in range (self.client)]
        layer_size = np.array([2,1])
        Layer_scale = [None for i in range (self.client)]
        j = 0
        for i in range(self.client):
            Layer_scale[i] = action[j:j+(self.args.n_layers+1):1]
            j = i + self.args.n_layers
        for client in range(self.client):
            Sampling[client] = Layer_scale[i]

        # process local GCN
        for i in range (self.client):
            self.Model[i], Time_cost[i] = runGCN(self.args, Train_info, self.GCN_info[i], Sampling[i])

        # federated learning
        Para_model = ModelAggregation(self.Model, self.global_model)
        for i in range (self.client):
            self.Model[i].load_state_dict(Para_model)
        self.global_model.load_state_dict(Para_model)

        # test global model
        time_cost = max(Time_cost)
        self.dur.append(time_cost)
        Acc = testGCN(self.args, Train_info, self.global_GCN_info, time_cost, self.dur)

        # calculate reward by 64^(acc - target) - 1
        reward = pow(64,(Acc-self.target)) - 2
        # reward = pow(64,(Acc-self.target)) - 10*time_cost

        # get the next state
        parm_local = {}
        S_local = [None for i in range (self.client)]

        for i in range (self.client):
            S_local[i] = []
            Name = []
            for name, parameters in self.Model[i].named_parameters():
                parm_local[name]=parameters.detach().cpu().numpy()
                Name.append(name)
            for j in range(len(Name)):
                for a in parm_local[Name[j]][0::].flatten():
                    S_local[i].append(a)
            S_local[i] = np.array(S_local[i]).flatten()
        # to 1-axis

        # convert to [num_samples, num_features]
        if self.dataset == 'cora': S = np.reshape(S_local,(self.client, 23063))
        if self.dataset == 'citeseer': S = np.reshape(S_local,(self.client, 59366))
        if self.dataset == 'pubmed': S = np.reshape(S_local,(self.client, 8611))

        # pca
        state = self.pca.fit_transform(S)
        state = state.flatten()

        # done?
        done = False
        if Acc >= self.target:
            done = True
        return state, reward, Acc, time_cost, done

    def reset(self):
        # local graph generate
        Local_graph = GraphPartition(self.args, self.client)
        for i in range (self.client):
            self.Graph[i] = Local_graph[i][0]

        for i in range (self.client+1):
            # local model conduct
            if i < self.client:
                self.GCN_info[i] = ModelConduct(self.args, Local_graph[i], 'client', i)
                self.Model[i] = self.GCN_info[i][0]
            # global model conduct

        self.global_GCN_info = ModelConduct(self.args, Local_graph[0], 'server', 100)
        self.global_model = self.global_GCN_info[0]

        parm_local = {}
        S_local = [None for i in range (self.client)]

        for i in range (self.client):
            S_local[i] = []
            Name = []
            for name, parameters in self.Model[i].named_parameters():
                # print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                Name.append(name)
            for j in range(len(Name)):
                for a in parm_local[Name[j]][0::].flatten():
                    S_local[i].append(a)
            S_local[i] = np.array(S_local[i]).flatten()
        # to 1-axis

        # convert to [num_samples, num_features]
        # S = np.reshape(S_local,(1,self.client*23335))
        if self.dataset == 'cora': S = np.reshape(S_local,(self.client, 23063))
        if self.dataset == 'citeseer': S = np.reshape(S_local,(self.client, 59366))
        if self.dataset == 'pubmed': S = np.reshape(S_local,(self.client, 8611))

        # pca
        state = self.pca.fit_transform(S)
        state = state.flatten()

        return state
        
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
