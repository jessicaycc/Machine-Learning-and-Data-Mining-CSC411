import os
import sys
import time
import torch
import random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


if not os.path.exists('ttt'):
    os.makedirs('ttt')

if not os.path.exists('plots'):
    os.makedirs('plots')

class Environment(object):
    '''
    The Tic-Tac-Toe Environment
    '''
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        '''Reset the game to an empty board.'''
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        '''Print what is on the board.'''
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        '''Check if someone has won the game.'''
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        '''Mark a point on position action.'''
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        '''Choose a random, unoccupied move on the board to play.'''
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        '''Play a move, and then have a random agent play the next move.'''
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError('???')
        return state, status, done

class Policy(nn.Module):
    '''
    The Tic-Tac-Toe Policy
    '''
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.features(x)


def select_action(policy, state):
    '''Samples an action from the policy at the state.'''
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=0.9):
    '''
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    '''
    returns = list()
    for i in range(len(rewards)):
        returns.append(sum(r*gamma**t for t,r in enumerate(rewards[i:])))

    return returns

def finish_episode(saved_rewards, saved_logprobs, gamma=0.9):
    '''Samples an action from the policy at the state.'''
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    '''Returns a numeric given an environment status.'''
    return {
            Environment.STATUS_VALID_MOVE  :  0,
            Environment.STATUS_INVALID_MOVE: -1,
            Environment.STATUS_WIN         :  1,
            Environment.STATUS_TIE         :  0,
            Environment.STATUS_LOSE        : -1
    }[status]

def first_move_distr(policy, env):
    '''Display the distribution of first moves.'''
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data

def load_weights(policy, episode):
    '''Load saved weights'''
    weights = torch.load('ttt/policy-%d.pkl' % episode)
    policy.load_state_dict(weights)


def play_self(env):
    # X - 1
    env.step(1)
    env.render()

    # O - 8
    env.step(8)
    env.render()

    # X - 6
    env.step(6)
    env.render()

    # O - 4
    env.step(4)
    env.render()

    # X - 0
    env.step(0)
    env.render()

    # O - 3
    env.step(3)
    env.render()

    # X - 2
    env.step(2)
    env.render()
    print('X wins!')

def train(policy, env, gamma=0.9, log_interval=1000):
    '''Train policy gradient.'''
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=gamma)
    running_reward = 0
    invalid_moves = 0
    average_return = []

    for i_episode in range(50000):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)

            if status == env.STATUS_INVALID_MOVE:
                invalid_moves += 1

            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}\tInvalid moves: {}'.format(
                i_episode,
                running_reward / log_interval,
                invalid_moves))
            print("First move distribution: \n", first_move_distr(policy, env).numpy(), "\n")
            average_return.append(running_reward / log_interval)
            running_reward = 0
            invalid_moves = 0
            

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       'ttt/policy-%d.pkl' % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    plt.plot(np.arange(0, 50000, 1000), average_return)
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.savefig('plots/learningCurve.png', bbox_inches='tight')
    plt.show()

def test(policy, env, ep, num_games=100, out=True):
    win, tie, lose = 0, 0, 0
    load_weights(policy, ep)

    for i in range(num_games):
        state = env.reset()
        done = False

        if i % 20 == 0 and out:
            print("Game", i)

        while not done:
            action, _ = select_action(policy, state)
            state, status, done = env.play_against_random(action)

            if i % 20 == 0 and out:
                env.render()

        if status == env.STATUS_WIN:
            win += 1
        elif status == env.STATUS_TIE:
            tie += 1
        elif status == env.STATUS_LOSE:
            lose += 1

    print('# Games: {}\tWins: {}\tTies: {} \tLosses: {}'.format(
        num_games, win, tie, lose))
    return win/num_games, tie/num_games, lose/num_games

def plot_performance(policy, env):
    win, tie, lose = list(), list(), list()

    for ep in range (0, 50000, 1000):
        w, t, l = test(policy, env, ep, out=False)
        win.append(w)
        tie.append(t)
        lose.append(l)

    plt.plot(np.arange(0, 50000, 1000), win, label='win')
    plt.plot(np.arange(0, 50000, 1000), tie, label='tie')
    plt.plot(np.arange(0, 50000, 1000), lose, label='lose')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.legend(loc='lower left')
    plt.savefig('plots/winRate.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    start = time.time()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    policy = Policy(hidden_size=96)
    env = Environment()

    play_self(env)
    train(policy, env)

    if len(sys.argv) > 1:
        # final model is policy-47000.pkl
        test(policy, env, int(sys.argv[1]))

    plot_performance(policy, env)

    end = time.time()
    print('Time elapsed: %.2fs' % (end-start))
