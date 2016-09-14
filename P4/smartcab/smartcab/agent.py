import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from q_learning import QLearning

# Global variables to track #actions and rewards during run
total_rewards = 0.
total_actions = 0.


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, gamma, alpha):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # Initialize additional variables and QLearning here
        self.actions = ['forward', 'left', 'right', None]
        self.states = []
        self.q_learn = QLearning(self.actions, states=self.states, gamma=gamma, alpha=alpha)
        self.p_action = None
        self.p_reward = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required

    def get_state_code(self, inputs):
        dic = {
            'red': 1,
            'green': 2,
            'forward': 1,
            'left': 2,
            'right': 3,
            None: 4
        }
        state_code = 0
        for i in xrange(len(inputs)):
            state_code += dic[inputs[i]] * 10**i
        return state_code

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Get previous state and action taken in state
        p_state, p_action, p_reward = self.state, self.p_action, self.p_reward

        # Get current state
        state = self.get_state_code([self.next_waypoint] + inputs.values())

        # if state is observed for the first time add it to Q and states list
        self.q_learn.add_new_state(state)

        # To avoid crash at first run when p_state == p_action == p_reward == None
        if p_state is not None:
            # Learn policy based on state, action, reward
            self.q_learn.update_Q(p_reward=p_reward, p_state=p_state, p_action=p_action, c_state=state)

        # Select action according to your policy
        action = self.q_learn.get_best_action(state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # sets previous state and action based for next state
        self.p_action, self.state, self.p_reward = action, state, reward

        global total_actions, total_rewards
        total_actions += 1
        total_rewards += reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, gamma=.8, alpha=.9)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print total_rewards
    print total_actions


if __name__ == '__main__':
    run()
