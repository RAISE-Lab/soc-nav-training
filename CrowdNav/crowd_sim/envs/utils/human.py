from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        """
        == From `Agent` ==

        Base class for robot and human. Have the physical attributes of an agent.
        config: info from config file
        section: section from config file -- i.e. 'human' or 'robot'
        """

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
