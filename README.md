### MuZero Homebrew

Homebrew implementation of MuZero using NumPy/PyTorch. I'll be writing more about the implementation [here](https://samgijsen.github.io/general/2022/12/13/MuZeroPart1/) once it gets further along.


The algorithm leverages the following pieces:
* `mcts.py`: Monte Carlo tree search, used to plan ahead by simulating game roll-outs.
* `models.py`: Includes the PyTorch models. 
  * **RepresentationNet**: $s_0 = h(o_t)$, encoding the current board position
  * **DynamicsNet**: $s_t+1, r = g(s_t, a_t)$, simulates a state transition but does so based on the embedding. Also predicts associated reward.
  * **PredictionNet**: $p_0, v_0 = f(s_0)$, can be queried for policy suggestion and value estimation of current state.
* `self_play.py`: Uses MCTS to run games and logs all activity for evaluation and training.
* `TTT_env.py`: Environment.

The resulting `Self_Play` and `Training` classes allow for a recursive process: first, a batch of games is played, which afterwards serve as training data for the models. This is a simple sequential example to get started:
```py

for v in range(iterations):
    
    # start with self-play
    print("Self-Play: Iter {} out of {}".format(v+1, iterations))


    engine = Self_Play(games=games, depth=depth, temperature=temperature, parameter_path = parameter_path)
    state_log, mcts_log, win_log, act_log, _ = engine.play(version=v, random_opponent=False)
    
    # train using the played games
    print("Train: Iter Net {} out of {}".format(v+1, iterations))
    
    if v == 0: # Initialization of training class
        train = Training(lr=lr, parameter_path=parameter_path, batchsize=batchsize, epochs=epochs, K=K, log_to_tensorboard=log_to_tensorboard)   
    nets, losses = train.train(state_log, mcts_log, win_log, act_log, version=v)
    
    # evaluation
    if eval_random:
        engine = Self_Play(games=eval_games, depth=depth, temperature=temperature, parameter_path = parameter_path)
        _, _, _, _, random_scores = engine.play(version=v+1, random_opponent=True)
        b_scores.append(random_scores)
        
    # record losses
    loss_p.append(losses["l_pol"])
    loss_r.append(losses["l_rew"])
    loss_v.append(losses["l_val"])
    loss_t.append(losses["loss"])
    
train.plot_losses(loss_p, loss_v, loss_r, loss_t)
    
```

