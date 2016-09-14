import random
import sys
import pandas as pd
import numpy as np


class QLearning:
    """Q-Learning Implementation"""

    # Initializes QLearning with given actions, states, alpha and gamma
    def __init__(self, actions, states, gamma, alpha):
        self.Q = pd.DataFrame(data=0.0, index=actions, columns=states)
        self.gamma = gamma
        self.alpha = alpha
        self.actions = actions
        self.states = states

    # return Q value of given state, action
    def get_Q(self, state, action):
        return self.Q[state][action]

    # adds state to Q when observed for first time
    def add_new_state(self, state):
        if state not in self.states:
            self.states.append(state)
            self.Q[state] = 0.0

    # returns all the the states observed
    def get_states(self):
        return self.states

    # Updates the Q values based on the state, action and reward
    def update_Q(self, p_reward, c_state, p_state, p_action):
        Q, alpha, gamma = self.Q, self.alpha, self.gamma
        Q[p_state][p_action] = Q[p_state][p_action] + alpha * (
            p_reward + gamma * np.argmax(Q[c_state][a] for a in self.actions) - Q[p_state][p_action]
        )

    # Make choice for best action in given state based on learning or take random action when observed of the first time
    def get_best_action(self, state):
        q = np.array([self.Q[state][x] for x in self.actions])
        if np.argmin(q) == np.argmax(q) < sys.float_info.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(q)]



