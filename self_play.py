import numpy as np
import torch
import pickle
import math

#from models import ValueNet, PolicyNet
from TTT_env import TTT
from mcts import MCTS, Node


class Self_Play():

    def __init__(self, parameter_path, games=1, depth=50, temperature=1, dir_alpha=1., dir_frac=0.25):

        self.games = games
        self.depth = depth
        self.temperature = temperature
        self.parameter_path = parameter_path
        self.dir_alpha = dir_alpha
        self.dir_frac = dir_frac

    def play(self, version, random_opponent):

        state_log = np.empty((0, 9), int)
        mcts_log = np.empty((0, 9), float)
        win_log = np.empty((0,1), int)
        act_log = np.empty((0,1), int)
        random_scores = [0,0,0]

        for i in range(self.games):

            state = [0]*9
            env = TTT()
            move = 0

            while env.check_terminality() == None:

                legal_moves = env.legal_moves(state)[0]

                if random_opponent and ((move % 2) == (i % 2)):

                    # random action
                    pi = np.ones(len(legal_moves))/len(legal_moves)
                    action = np.random.choice(np.arange(len(legal_moves), dtype="float64"), p=pi, size=1)[0].astype(int)
                
                else: 

                    # perform search
                    mcts = MCTS(state, turn=1, runs=self.depth, parameter_path=self.parameter_path, dir_alpha=self.dir_alpha, dir_frac=self.dir_frac)
                    root = mcts.search(version=version)

                    # select move
                    action, pi = root.sample_action(temperature=self.temperature)


                # log
                state_log = np.append(state_log, np.array(state).reshape(1,-1), axis=0)
                pi_masked = np.zeros(len(state))
                pi_masked[legal_moves] = pi 
                mcts_log = np.append(mcts_log, pi_masked.reshape(1,-1), axis=0)
                act_log = np.append(act_log, np.array(action).reshape(1,-1), axis=0)
 
                # make move
                state[action] = 1 

                # prep board for 'opposing' player
                state = [-x for x in state] 
                env = TTT(state)
                move += 1

            z = env.check_terminality()

            if z != 0: # not a draw
                if z == 1 and (move % 2) == 1: # 'beginner' won
                    random_scores[i%2] += 1 # index0 is random player win
                elif z == -1 and (move % 2) == 0: # 'beginner' won
                    random_scores[i%2] += 1
                    z = 1
                elif z == -1 and (move % 2) == 1: # beginner lost
                    random_scores[1-(i%2)] += 1
                elif z == 1 and (move % 2) == 0: # beginner lost
                    random_scores[1-(i%2)] += 1
                    z = -1
            else:
                random_scores[2] += 1


            for t in range(move):
                if t == 0: # code first state always as a'draw' ending - TODO: exclude from training
                    win_log = np.append(win_log, np.array(0).reshape(1, -1), axis=0)
                else:
                    win_log = np.append(win_log, np.array(z).reshape(1, -1), axis=0)
                z *= -1

        if not random_opponent:
            self.save_game_data(version, self.parameter_path, state_log, mcts_log, win_log, act_log)

        return state_log, mcts_log, win_log, act_log, random_scores


    def save_game_data(self, version, parameter_path, state_log, mcts_log, win_log, act_log):

        fn = parameter_path + "game_data_v{}".format(version) + ".data"

        # increase our training data by augmenting game states
        # 90, 180, 270 degree rotations, vertical and horizontal mirrors

        l = state_log.shape[0] # length by 9

        # # guides to transform actions
        guide = np.arange(9).reshape(3,3)
        guide_rot90 = np.rot90(guide)
        guide_rot180 = np.rot90(guide_rot90)
        guide_rot270 = np.rot90(guide_rot180)
        guide_fliplr = np.fliplr(guide)
        guide_flipud = np.flipud(guide)
        
        for i in range(l):

            s = state_log[i,:].reshape(3,3)
            state_log = np.append(state_log, np.rot90(s).reshape(1,-1), axis=0)
            state_log = np.append(state_log, np.rot90(np.rot90(s)).reshape(1,-1), axis=0)
            state_log = np.append(state_log, np.rot90(np.rot90(np.rot90(s))).reshape(1,-1), axis=0)
            state_log = np.append(state_log, np.fliplr(s).reshape(1,-1), axis=0)
            state_log = np.append(state_log, np.flipud(s).reshape(1,-1), axis=0)

            s = mcts_log[i,:].reshape(3,3)
            mcts_log = np.append(mcts_log, np.rot90(s).reshape(1,-1), axis=0)
            mcts_log = np.append(mcts_log, np.rot90(np.rot90(s)).reshape(1,-1), axis=0)
            mcts_log = np.append(mcts_log, np.rot90(np.rot90(np.rot90(s))).reshape(1,-1), axis=0)
            mcts_log = np.append(mcts_log, np.fliplr(s).reshape(1,-1), axis=0)
            mcts_log = np.append(mcts_log, np.flipud(s).reshape(1,-1), axis=0)

            for j in range(5):
                win_log = np.append(win_log, win_log[i].reshape(1,-1), axis=0)

            a = act_log[i]
            act_log = np.append(act_log, np.where(guide_rot90 == a)[0][0].reshape(1,-1), axis=0)
            act_log = np.append(act_log, np.where(guide_rot180 == a)[0][0].reshape(1,-1), axis=0)
            act_log = np.append(act_log, np.where(guide_rot270 == a)[0][0].reshape(1,-1), axis=0)
            act_log = np.append(act_log, np.where(guide_fliplr == a)[0][0].reshape(1,-1), axis=0)
            act_log = np.append(act_log, np.where(guide_flipud == a)[0][0].reshape(1,-1), axis=0)

        with open(fn, "wb") as f:
            pickle.dump([state_log, mcts_log, win_log, act_log], f)
