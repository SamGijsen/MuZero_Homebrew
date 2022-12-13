import numpy as np
import torch
import torch.nn.functional as F
from models import RepresentationNet, PredictionNet, DynamicsNet
from TTT_env import TTT
import math


class MCTS():
    
    def __init__(self, state, turn, runs, parameter_path, dir_alpha=1., dir_frac=0.25):
               
        self.state = state
        self.turn = turn
        self.runs = runs
        self.parameter_path = parameter_path
        self.dir_alpha = dir_alpha
        self.dir_frac = dir_frac

    def search(self, version):

        torch.manual_seed(version)
        
        representation = RepresentationNet()
        prediction = PredictionNet()
        dynamics = DynamicsNet()

        if version > 0: # load parameters from previous versions
            representation.load_state_dict(torch.load(self.parameter_path + "representation_v{}".format(version-1)))
            prediction.load_state_dict(torch.load(self.parameter_path + "prediction_v{}".format(version-1)))
            dynamics.load_state_dict(torch.load(self.parameter_path + "dynamics_v{}".format(version-1)))

        # determine legal moves
        env = TTT(self.state, turn=self.turn) 
        legal_moves = env.legal_moves(self.state)[0]

        # embed state
        embedded_state = representation(torch.tensor(self.state).float()).view(1, representation.l3.out_features) # turn off grad?
        
        # expand the root
        p_policy, _ = prediction.predict(embedded_state) 
        p_policy = p_policy.squeeze()
        p_policy[np.isin(np.arange(9), legal_moves, invert=True)] = 0 # mask illegal moves
        p_policy /= np.sum(p_policy) # normalize
    
        root = Node(embedded_state, self.turn, legal_moves, self.dir_alpha, self.dir_frac)
        root.expand(self.turn, prior=p_policy.copy(), dynamics=dynamics)

        # do MCTS steps 
        for run in range(self.runs):
            node_t = root
            search_path = [node_t]

            while node_t.expanded:
                node_t = self.select_child(node_t)
                search_path.append(node_t)

            p_policy, Q = prediction.predict(node_t.embedded_state)
            p_policy = p_policy.squeeze()
            node_t.expand(node_t.turn, prior=p_policy, dynamics=dynamics)

            # backup
            for i in range(len(search_path)-1, -1, -1):
                search_path[i].value += Q*search_path[i].turn
                search_path[i].N += 1

        return root


    def UCB_scoring(self, node):

        ucb_scores = []

        for a in range(len(node.child)):

            #score = 3 * node.child[a].prior * math.sqrt(node.N) / (node.child[a].N+1) # AZ
            score = node.child[a].prior * (math.sqrt(node.N) / (node.child[a].N+1)) * (1.25 + np.log(node.N+19652+1)/(19652)) # MZ
            if node.child[a].N > 0:
                v = -node.child[a].value / node.child[a].N
            else:
                v = 0
            ucb_scores.append(v + score)

        return ucb_scores


    def select_child(self, node):

        scores = self.UCB_scoring(node)
        a = np.argmax(scores)

        return node.child[a]


class Node():
    
    def __init__(self, embedded_state, turn, legal_moves, dir_alpha=1., dir_frac=0.25):
        
        self.N = 0 # visits
        self.prior = 0
        self.child = {}
        self.embedded_state = embedded_state
        self.expanded = False
        self.value = 0
        self.turn = turn
        self.legal_moves = legal_moves
        self.dir_alpha = dir_alpha
        self.dir_frac = dir_frac

    def expand(self, turn, prior, dynamics):

        self.expanded = True
        # 1. generate child
        # 2. assign priors

        prior = prior[self.legal_moves]

        # add Dirichlet-noise
        noise = np.random.dirichlet([self.dir_alpha]*len(prior))
        prior = prior*(1-self.dir_frac) + noise*self.dir_frac
        prior /= np.sum(prior)

        for i, a in enumerate(prior):

            a_oh = F.one_hot(torch.tensor(self.legal_moves[i]), num_classes=9).float()
            next_state, _ = dynamics.predict(torch.cat([self.embedded_state, a_oh.view(1,-1)], dim=1)) 
            self.child[i] = Node(torch.tensor(next_state), turn*-1, self.legal_moves, self.dir_alpha, self.dir_frac)
            self.child[i].prior = prior[i].item()

    def sample_action(self, temperature=1): 

        visits = [self.child[i].N**temperature for i in range(len(self.child))]
        pi = np.array(visits, dtype="float64")/np.sum(visits)
        a = np.random.choice(np.arange(len(visits), dtype="float64"), p=pi, size=1)[0]
        
        return self.legal_moves[a.astype(int)], pi
