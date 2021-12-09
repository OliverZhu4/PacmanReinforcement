# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            _val = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    self.values[state] = self.mdp.getReward(state, 'exit', '')
                else:
                    maxx = None
                    for it in self.mdp.getPossibleActions(state):
                        if maxx is None:
                            maxx = self.computeQValueFromValues(state, it)
                        else:
                            maxx = max(maxx, self.computeQValueFromValues(state, it))
                    _val[state] = maxx # assign the maxx back to the values
            self.values = _val

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        v = 0
        for vk in self.mdp.getTransitionStatesAndProbs(state, action):
            Rsas = self.mdp.getReward(state, action, vk[0])
            v += Rsas + self.discount * self.values[vk[0]] * vk[1]
        return v

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action_dict = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            action_dict[action] = self.computeQValueFromValues(state, action)
        ans = action_dict.argMax() # return the arg max, remember to carefully check the condition not to throw an error
        return ans

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration a gent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        ans = 0
        old_state = self.mdp.getStates()
        for i in range(self.iterations):
            if ans == len(old_state):
                ans = 0
            action = self.getAction(old_state[ans])
            if action is not None:
                self.values[old_state[ans]] = self.getQValue(old_state[ans], action)
            else:
                self.values[old_state[ans]] = 0
            ans += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        q = util.PriorityQueue()
        prev_value = self.values
        states = self.mdp.getStates()
        predecessors = dict()

        for state in states:
            predecessors[state] = set()

        for state in states:
            values = util.Counter()
            cur_value = prev_value[state]
            for k in self.mdp.getPossibleActions(state):
                for it in self.mdp.getTransitionStatesAndProbs(state, k):
                    if it[1] != 0:
                        predecessors[it[0]].add(state)
                values[k] = self.computeQValueFromValues(state, k)
                diff = abs(cur_value - values[values.argMax()])
                q.update(state, -diff)
        cnt = 0

        while cnt in range(self.iterations) and not q.isEmpty():
            cnt += 1
            state = q.pop()
            if not self.mdp.isTerminal(state):
                q_val = util.Counter()
                for it in self.mdp.getPossibleActions(state):
                    q_val[it] = self.computeQValueFromValues(state, it)
                prev_value[state] = q_val[q_val.argMax()]

            for p in predecessors[state]:
                q_val = util.Counter()
                # get the possible actions
                for it in self.mdp.getPossibleActions(p):
                    q_val[it] = self.computeQValueFromValues(p, it)
                diff = abs(prev_value[p] - q_val[q_val.argMax()])
                if diff > self.theta:
                    q.update(p, -diff)

