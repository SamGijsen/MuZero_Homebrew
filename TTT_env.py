import numpy as np

class TTT():
    
    def __init__(self, state="reset", turn=1):
        
        if state == "reset":
            self.reset()
        else:
            self.state = state
            self.turn = turn
            
            if self.win(self.state):
                self.done = True
            else:
                self.done = False

        
    def reset(self):
        self.done = False
        self.state = [0]*9
        self.turn = 1
        
    def step(self, a, turn):
        
        r = 0 
        if self.state[a] != 0: # Illegal move
            r = -1
            self.done = True
        else:
            self.state[a] = turn
        
        if self.win(self.state): # Check for win
            r = 1
            self.done = True
            
        return r
            
    def win(self, s):
        for r in [0, 3, 6]: # horizontals
            if (s[r]==self.turn) and (s[r]==s[r+1]==s[r+2]):
                return True
        for r in [0,1,2]: # verticals
            if (s[r]==self.turn) and (s[r]==s[r+3]==s[r+6]):
                return True   
        if (s[0]==self.turn) and (s[0]==s[4]==s[8]): 
            return True # diagonal
        if (s[2]==self.turn) and (s[2]==s[4]==s[6]): 
            return True # diagonal

    def check_terminality(self):
        if self.win(self.state):
            return 1 # win
        self.turn = -self.turn 
        if self.win(self.state):
            return -1 # loss
        if self.state.count(0) == 0:
            return 0 # draw
        return None # no terminal state

    def legal_moves(self, state):
        return np.argwhere(np.array(state)==0).reshape(1,-1)

    def illegal_moves(self, state):
        return np.argwhere(np.array(state)!=0).reshape(1,-1)

