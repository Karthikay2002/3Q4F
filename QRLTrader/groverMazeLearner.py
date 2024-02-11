import datetime as dt
from math import ceil

import numpy as np
from marketsimcode import compute_portvals
from qiskit import *
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector


class GroverMazeLearner:
    """
    Inits a quantum QLearner object for given environment.
    Environment must be discrete and of "maze type", with the last state as the goal
    """

    def __init__(
        self,
        learner,
        num_states,
        num_actions=3,
        alpha=0.2,
        gamma=0.6,
    ):  # gym.make("FrozenLake-v0", is_slippery=False)
        # state and action spaces dims
        self.obs_dim = num_states
        self.acts_dim = num_actions
        self.env = learner
        # dim of qubits register needed to encode all actions
        self.acts_reg_dim = ceil(np.log2(self.acts_dim))
        # optimal number of steps in original Grover's algorithm
        self.max_grover_steps = int(
            round(np.pi / (4 * np.arcsin(1.0 / np.sqrt(2**self.acts_reg_dim))) - 0.5)
        )
        # quality values
        self.state_vals = np.zeros(self.obs_dim)
        # grover steps taken
        self.grover_steps = np.zeros((self.obs_dim, self.acts_dim), dtype=int)
        # boolean flags to signal maximum amplitude amplification reached
        self.grover_steps_flag = np.zeros((self.obs_dim, self.acts_dim), dtype=bool)
        # learner hyperparms (eps still not used)
        self.hyperparams = {
            "k": -1,
            "alpha": alpha,
            "gamma": gamma,
            "eps": 0.01,
            "max_epochs": 50,
            "max_steps": 50,
            "graphics": True,
        }
        # current state
        # self.state = self.env.return_state(0)
        # current action
        self.action = 0
        # list of grover oracles
        self.grover_ops = self._init_grover_ops()
        # list of state-action circuits
        self.acts_circs = self._init_acts_circs()
        # qiskit simulator
        self.SIM = Aer.get_backend("qasm_simulator")

    def set_hyperparams(self, hyperdict):
        """
        Set learner's hyperparameters
        :param hyperdict: a dict with same keys as self's
        :return:
        """
        self.hyperparams = hyperdict

    def _init_acts_circs(self):
        """
        Inits state-action circuits
        :return: list of qiskit circuits, initialized in full superposition
        """
        circs = [
            QuantumCircuit(self.acts_reg_dim, name="|as_{}>".format(i))
            for i in range(self.obs_dim)
        ]
        for c in circs:
            c.h(list(range(self.acts_reg_dim)))
        return circs

    def _update_statevals(self, reward, new_state):
        """
        Bellman equation for state values update
        :param reward: instantaneous reward received by the agent
        :param new_state: state reached upon taking previous action
        :return:
        """
        self.state_vals[self.state] += self.hyperparams["alpha"] * (
            reward
            + self.hyperparams["gamma"] * self.state_vals[new_state]
            - self.state_vals[self.state]
        )

    def _eval_grover_steps(self, reward, new_state):
        """
        Choose how many grover step to take based on instantaneous reward and value of new state
        :param reward: the instantaneous reward received by the agent
        :param new_state: the new state visited by the agent
        :return: number of grover steps to be taken,
        if it exceeds the theoretical optimal number the latter is returned instead
        """
        steps_num = int(self.hyperparams["k"] * (reward + self.state_vals[new_state]))
        return min(steps_num, self.max_grover_steps)

    def _init_grover_ops(self):
        """
        Inits grover oracles for the actions set
        :return: a list of qiskit instructions ready to be appended to circuit
        """
        states_binars = [
            format(i, "0{}b".format(self.acts_reg_dim)) for i in range(self.acts_dim)
        ]
        targ_states = [Statevector.from_label(s) for s in states_binars]
        grops = [GroverOperator(oracle=ts) for ts in targ_states]
        return [g.to_instruction() for g in grops]

    def _run_grover(self):
        """
        DEPRECATED
        :return:
        """
        # deploy grover ops on acts_circs
        gsteps = self.grover_steps[self.state, self.action]
        circ = self.acts_circs[self.state]
        op = self.grover_ops[self.action]
        for _ in range(gsteps):
            circ.append(op, list(range(self.acts_reg_dim)))
        self.acts_circs[self.state] = circ

    def _run_grover_bool(self):
        """
        Update state-action circuits based on evaluated steps
        :return:
        """
        flag = self.grover_steps_flag[self.state, :]
        gsteps = self.grover_steps[self.state, self.action]
        circ = self.acts_circs[self.state]
        op = self.grover_ops[self.action]
        if not flag.any():
            for _ in range(gsteps):
                circ.append(op, list(range(self.acts_reg_dim)))
        if gsteps >= self.max_grover_steps and not flag.any():
            self.grover_steps_flag[self.state, self.action] = True
        self.acts_circs[self.state] = circ

    def return_action(self, state):
        self.state = state
        return self._take_action()

    def _take_action(self):
        """
        Measures the state-action circuit corresponding to current state and decides next action
        :return: action to be taken, int
        """
        circ = self.acts_circs[self.state]
        circ_tomeasure = circ.copy()
        circ_tomeasure.measure_all()
        # circ_tomeasure = transpile(circ_tomeasure)
        # print(circ.draw())
        job = execute(circ_tomeasure, backend=self.SIM, shots=1)
        result = job.result()
        counts = result.get_counts()
        action = int((list(counts.keys()))[0], 2)
        return action

    def train(self):
        """
        groverize and measure action qstate -> take corresp action
        obtain: newstate, reward, terminationflag
        update stateval, grover_steps
        for epoch in epochs until max_epochs is reached
        :return:
        dictionary of trajectories
        """
        # traj_dict = {}

        # set initial max_steps
        # optimal_steps = self.hyperparams["max_steps"]
        l = []
        for epoch in range(self.hyperparams["max_epochs"]):
            if epoch % 1 == 0:
                print("Processing epoch {} ...".format(epoch))
                # reset env
            self.env.m = 0
            self.state = self.env.return_state(0)
            # init list for traj
            traj = [self.state]

            for step in range(len(self.env.y)):
                # print("Taking step {0}/{1}".format(step, optimal_steps), end="\r")
                # print('STATE: ', self.state)
                # Select action
                self.action = self._take_action()
                # take action
                # print(self.action)
                if self.action >= 3:
                    continue
                else:
                    new_state, reward = self.env.get_reward(self.action, step)
                # print('REWARD: ', reward)
                # update statevals and grover steps
                self._update_statevals(reward, new_state)
                self.grover_steps[self.state, self.action] = self._eval_grover_steps(
                    reward, new_state
                )
                # amplify amplitudes with grover
                # self._run_grover()
                self._run_grover_bool()
                # render if curious
                # save transition
                traj.append(new_state)
                # quit epoch if done
                # move to new state
                self.state = new_state
                # print('STATE_VALS: ', self.state_vals)
                # print('GROVER_STEPS: ', self.grover_steps)

            # traj_dict["epoch_{}".format(epoch)] = traj
            self.env.m = 0
            trades = self.env.testPolicy(sd=self.env.sd, ed=self.env.ed)
            portvals = compute_portvals(
                trades,
                self.env.symbol,
                start_val=100000,
                start_date=self.env.sd,
                end_date=self.env.ed,
            )
            # print(len(trades))
            cr = portvals.iloc[-1, 0] / portvals.iloc[0, 0] - 1
            print(cr)
            j = 0
            while j < len(l):
                if l[j] < cr:
                    break
                j = j + 1
            l.insert(j, cr)
            if len(l) > 20 and j < 0.2 * len(l):
                print("Stopped at " + str(cr))
                break
