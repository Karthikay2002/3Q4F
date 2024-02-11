import random as rand

import numpy as np


class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = None
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.rar = rar
        self.radr = radr
        self.QTable = np.zeros((num_states, num_actions))
        if self.dyna != 0:
            self.R = np.zeros((num_states, num_actions))
            self.T_c = np.zeros((num_states, num_actions, num_states))
            self.T_c.fill(0.000001)
            self.T = self.T_c / np.sum(self.T_c, axis=2, keepdims=True)
            self.A = 0.75

    def author(self):
        return "karcot3"

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        self.s = s
        action = self.QTable[s, :].argmax()
        self.a = action
        return action

    def perform_dyna(self, dyna):
        state_list = np.random.randint(0, self.num_states, size=dyna)
        action_list = np.random.randint(0, self.num_actions, size=dyna)
        new_states = np.array(
            [
                np.random.multinomial(1, self.T[s1][a1]).argmax()
                for s1, a1 in zip(state_list, action_list)
            ]
        )
        new_actions = self.QTable.argmax(axis=1)
        for s1, a1, s2 in zip(state_list, action_list, new_states):
            a2 = new_actions[s2]
            self.QTable[s1][a1] += self.alpha * (
                self.R[s1][a1] + self.gamma * self.QTable[s2][a2] - self.QTable[s1][a1]
            )

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        a2 = self.QTable[s_prime, :].argmax()
        self.QTable[self.s][self.a] += self.alpha * (
            r + self.gamma * self.QTable[s_prime][a2] - self.QTable[self.s][self.a]
        )
        action = self.QTable[s_prime, :].argmax()
        p = rand.random()
        if p < self.rar:
            action = rand.randrange(self.num_actions)
        self.rar = self.rar * self.radr
        if self.dyna != 0:
            self.T_c[self.s][self.a][s_prime] += 1
            self.T = self.T_c / np.sum(self.T_c, axis=2, keepdims=True)
            self.R[self.s][self.a] = (1 - self.A) * self.R[self.s][self.a] + self.A * r
            self.perform_dyna(self.dyna)
        self.s = s_prime
        self.a = action
        return action


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
