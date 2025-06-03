import numpy as np

class env_SL_4nd(object, ):
    def __init__(self, state_length=10):
        self.agent_number = 4
        self.action_space = 2
        self.state_space = 8
        self.M = state_length

    def reset(self):  # Initialize and return link_obs
        init_state = [np.zeros([self.state_space * self.M, ])] * self.agent_number  # agent_number basic states
        return init_state

    def D2O(self, action):
        '''
        Convert 2-bit one-hot action to 1-bit action:
        [1,0] => 0, [0,1] => 1
        '''
        action_o = 0
        if action[0] == 1:
            action_o = 0
        elif action[1] == 1:
            action_o = 1

        return action_o

    def step(self, actions):
        '''
        Take actions and return link_obs, reward, done, _
        actions: a list of action arrays for each agent
        link_obs encoding:
            own successful transmission: [1,0,0,0]
            other successful transmission: [0,1,0,0]
            collision: [0,0,1,0]
            idle: [0,0,0,1]
        '''
        # Initialize
        action1 = actions[0]
        action2 = actions[1]
        action3 = actions[2]
        action4 = actions[3]
        # reward records transmission status on the link
        reward = np.zeros([self.agent_number, ])

        link_obs = [np.zeros([4, ])] * self.agent_number
        action1_o, action2_o, action3_o, action4_o = self.D2O(action1), self.D2O(action2), self.D2O(action3), self.D2O(action4)

        total_action = action1_o + action2_o + action3_o + action4_o
        l_collision = np.zeros([self.agent_number, ])  # record collisions
        # print(total_action)
        if total_action == 0:  # link idle
            # link_obs
            link_obs = [[0, 0, 0, 1]] * self.agent_number
        elif total_action == 1:  # successful transmission on link
            if action1_o == 1:
                reward[0] = 1
                link_obs[0] = [1, 0, 0, 0]
                link_obs[1], link_obs[2], link_obs[3] = [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]
            elif action2_o == 1:
                reward[1] = 1
                link_obs[1] = [1, 0, 0, 0]
                link_obs[0], link_obs[2], link_obs[3] = [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]
            elif action3_o == 1:
                reward[2] = 1
                link_obs[2] = [1, 0, 0, 0]
                link_obs[0], link_obs[1], link_obs[3] = [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]
            elif action4_o == 1:
                reward[3] = 1
                link_obs[3] = [1, 0, 0, 0]
                link_obs[1], link_obs[2], link_obs[0] = [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]

        else:  # collision on link
            link_obs = [[0, 0, 1, 0]] * self.agent_number
            if action1_o == 1:
                l_collision[0] = 1
            if action2_o == 1:
                l_collision[1] = 1
            if action3_o == 1:
                l_collision[2] = 1
            if action4_o == 1:
                l_collision[3] = 1

        # self.update()
        return link_obs, reward, l_collision  # , done