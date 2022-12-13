import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt


class RepresentationNet(torch.nn.Module):
    """s_0 = h(o_t)"""

    def __init__(self):
        super(RepresentationNet, self).__init__()
        self.l1 = nn.Linear(9,32)
        self.l2 = nn.Linear(32,32)
        self.l3 = nn.Linear(32,32)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = torch.tanh(x)
        return x

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            p = self.forward(state)

        return p.cpu().numpy()


class DynamicsNet(torch.nn.Module):
    """s_{t+1}, r = g(s_t, a_t)""" 

    def __init__(self):
        super(DynamicsNet, self).__init__()
        self.l1 = nn.Linear(41,32)
        self.l2 = nn.Linear(32,32)

        # two heads
        self.state_head = nn.Linear(32,32)
        self.reward_head = nn.Linear(32,1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)

        state_next = torch.tanh(self.state_head(x))
        reward = torch.tanh(self.reward_head(x))
 
        return state_next, reward

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            s, r = self.forward(x)

        return s.cpu().numpy(), r.cpu().numpy()

class PredictionNet(torch.nn.Module):
    """p, v = f(s)"""

    def __init__(self):
        super(PredictionNet, self).__init__()
        self.l1 = nn.Linear(32,32)
        self.l2 = nn.Linear(32,32)
        self.policy_head = nn.Linear(32,9)
        self.value_head = nn.Linear(32,1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)

        policy = F.softmax(self.policy_head(x),dim=1)
        value = torch.tanh(self.value_head(x))

        return policy, value

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            p = self.forward(state)

        return p[0].cpu().numpy(), p[1].cpu().numpy()

    def CrossEntropy(self, output, y):
        return -(y * torch.log(output)).sum(dim=1)


class Training():

    def __init__(self, parameter_path, lr=0.02, batchsize=32, epochs=10, K=3, log_to_tensorboard=False):

        self.parameter_path = parameter_path
        self.lr = lr
        self.batchsize = batchsize
        self.epochs = epochs
        self.K = K
        self.log_to_tensorboard = log_to_tensorboard
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(f'logs/net')

        print("Device=", self.device)

    def train(self, state_log, mcts_log, win_log, act_log, version):

        self.check_inputs()

        dynamics = DynamicsNet().to(self.device)
        prediction = PredictionNet().to(self.device)
        representation = RepresentationNet().to(self.device)

        if version > 0: # load parameters from previous versions
            representation.load_state_dict(torch.load(self.parameter_path + "representation_v{}".format(version-1)))
            prediction.load_state_dict(torch.load(self.parameter_path + "prediction_v{}".format(version-1)))
            dynamics.load_state_dict(torch.load(self.parameter_path + "dynamics_v{}".format(version-1)))

        MSE_loss = nn.MSELoss()
        opt = torch.optim.SGD(list(dynamics.parameters()) + list(prediction.parameters()) + list(representation.parameters()), 
        lr=self.lr, weight_decay=1e-5)
        
        l_pol, l_val, l_rew = [], [], []
        loss = []

        for i in trange( int(len(win_log)*self.epochs/self.batchsize) ):

            idx = np.random.randint(0, high=len(win_log)-1-self.K, size=self.batchsize) #-self.K

            o_t = torch.tensor(state_log[idx,:]).float().requires_grad_()
            z_t = torch.tensor(win_log[idx,:]).float().requires_grad_()
            o_t, z_t = o_t.to(self.device), z_t.to(self.device)
            
            opt.zero_grad()

            st = representation(o_t)
            p,v,r = [],[],[]
            pt, vt = prediction(st)
            p.append(pt), v.append(vt)

            idx_absorb = idx.copy() # indices corrected for absorbing states
            id_k_list = [idx.copy()] # list of indices for each step k
            NAB_list = [] # records whether a state is absorbing (inverse Boolean for easy indexing) 

            for k in range(self.K):
                
                # check for absorbing states
                not_absorb = ((state_log[idx+1+k,:]==0).sum(axis=1) != 9) 
                
                if k == 0:
                    idx_bool_either = not_absorb.copy()
                else: # after k=0 we need to check if current or any preceding state is absorbing
                    idx_bool_either = [ x and y for (x,y) in zip(not_absorb, NAB_list[-1])] 
                
                idx_absorb[idx_bool_either] += 1 # indices for non-absorbing states are incremented by 1

                NAB_list.append(idx_bool_either.copy())
                id_k_list.append(idx_absorb.copy())

                a_t = F.one_hot(torch.tensor(act_log[id_k_list[k]]), num_classes=9).float().requires_grad_()
                a_t = a_t.to(self.device)

                st, rt = dynamics(torch.cat([st,a_t.view(self.batchsize, 9)],dim=1))
                r.append(rt)

                pt, vt = prediction(st)
                p.append(pt), v.append(vt)

            # compute losses
            lp,lv,lr = 0,0,0

            pi_t = torch.tensor(mcts_log[idx,:]).float().requires_grad_()
            z_t = torch.tensor(win_log[idx,:]).float().requires_grad_()
            pi_t, z_t = pi_t.to(self.device), z_t.to(self.device)

            lp += prediction.CrossEntropy(p[0], pi_t).mean()

            lv += MSE_loss(v[0], z_t)

            for k in range(self.K):
                
                pi_t = torch.tensor(mcts_log[id_k_list[k+1],:]).float().requires_grad_()
                z_t = torch.tensor(win_log[id_k_list[k+1],:]).float().requires_grad_()
                pi_t, z_t = pi_t.to(self.device), z_t.to(self.device)

                lp += prediction.CrossEntropy(p[k+1], pi_t).mean()
                lv += MSE_loss(v[k+1], z_t) 
                
                # if k > 0:
                #     lr += MSE_loss(r[k+1], z_t)  

            l = lp + lv + lr# total loss 

            l.backward()

            opt.step()

            l_pol.append(lp.item())
            l_val.append(lv.item())
            #l_rew.append(lr.item() if K>1 else 0)
            loss.append(l.item())

            if self.log_to_tensorboard and i%100==0:
                self.writer.add_scalar('Loss', l.item(), version*(int(len(win_log)*self.epochs/self.batchsize)) + i)
                self.writer.add_scalar('Policy Loss', lp.item(), version*(int(len(win_log)*self.epochs/self.batchsize)) + i)
                self.writer.add_scalar('Value Loss', lv.item(), version*(int(len(win_log)*self.epochs/self.batchsize)) + i)
                self.writer.add_scalar('Reward Loss', lr, version*(int(len(win_log)*self.epochs/self.batchsize)) + i)

        losses = {"l_pol": l_pol, "l_val": l_val, "l_rew": l_rew, "loss": loss}
        nets = [dynamics, representation, prediction]

        self.save_param_loss(version, nets, losses, ["dynamics", "representation", "prediction"])

        return nets, losses

    def CrossEntropy(self, output, y):

        return -(y * torch.log(output)).sum(dim=1)


    def save_param_loss(self, version, model, losses, net_prefix):

        loss_fn = self.parameter_path + "loss_v{}".format(version) + ".data"
        with open(loss_fn, "wb") as f:
            pickle.dump(losses, f) # does dict saving work?

        # Save parameters
        for i in range(len(model)):
            p_fn = self.parameter_path + net_prefix[i] + "_v{}".format(version)
            torch.save(model[i].state_dict(), p_fn)


    def check_inputs(self):
        
        if self.K<1:
            raise ValueError("K must be >= 1")


    def plot_losses(self, loss_p, loss_v, loss_r, loss_t):
    
        fig, ax = plt.subplots(2,2,figsize=(14,6))

        c = 0
        for i, l in enumerate(loss_p):
            ax[0,0].plot(np.arange(c, c+len(l)), loss_p[i], label="Iter {}".format(i),alpha=.8)
            c += len(l)

        ax[0,0].set_title("Policy loss")

        c = 0
        for i, l in enumerate(loss_v):
            ax[0,1].plot(np.arange(c, c+len(l)), loss_v[i], label="Iter {}".format(i))
            c += len(l)

        ax[0,1].set_title("Value loss")

        ax[1,0].set_title("Reward loss")

        c = 0
        for i, l in enumerate(loss_t):
            ax[1,1].plot(np.arange(c, c+len(l)), loss_t[i], label="Iter {}".format(i))
            c += len(l)

        ax[1,1].set_title("Total loss")
        
        plt.tight_layout()
        plt.show()
        
        return #fig, ax