import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm

from env_singlelink import env_SL_4nd  # Import environment from another file


def onehot_from_logits(logits, eps=0.01):
    ''' Generate one-hot representation of optimal action (implementation details, no need to fully understand) '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # Generate random action and convert to one-hot
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # Select action using epsilon-greedy
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """
    Sample from Gumbel(0,1) distribution
    :param shape: Action space dimension
    :param eps: Prevents log(0) errors
    :param tens_type: Tensor type
    :return: Gumbel sample (gi in formula)
    (Implementation details, no need to fully understand)
    """
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)  # Uniform(0,1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Sample from Gumbel-Softmax distribution (implementation details) """
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)  # Add Gumbel noise to logits
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """ Sample from Gumbel-Softmax and discretize (implementation details) """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)  # Convert to one-hot
    y = (y_hard.to(logits.device) - y).detach() + y  # Preserve gradients
    return y


class TwoLayerFC(torch.nn.Module):
    '''
    Build a 3-layer fully connected network for critic, 128 nodes
    '''

    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FCwithGRU(torch.nn.Module):
    '''
    Build a 4-layer network with GRU structure for actor, handles sequential tasks
    '''

    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.rnn = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)
        self.hidden = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x, self.hidden = self.rnn(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG algorithm for a single agent '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim_a, hidden_dim_c,
                 actor_lr, critic_lr, device):
        '''
        state_dim and action_dim are for actor network (per agent)
        state_dim = , input to actor network
        action_dim = 4, 4-bit one-hot encoding representing agent's actions on two links
        critic_input_dim: Aggregated information from all agents, input to critic
        actor_lr, critic_lr: Learning rates
        hidden_dim_a: 64
        hidden_dim_c: Critic network slightly wider than actor (128)
        '''

        # Actor-critic with target networks - total 4 networks
        self.actor = FCwithGRU(state_dim, action_dim, hidden_dim_a).to(device)
        self.target_actor = FCwithGRU(state_dim, action_dim,
                                      hidden_dim_a).to(device)

        # All agents share a centralized Critic network
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim_c).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim_c).to(device)

        # Initialize target networks with same weights
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        '''
        Take action based on state
        :param state: Current state
        :param explore: Whether to explore
        :return: Action array of dimension (4,) for link selection
        '''
        action = self.actor(state)  # Actor network converts state to action
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        ''' Soft update target networks '''
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim_a, hidden_dim_c,
                 state_dim, action_dim, critic_input_dim, gamma, tau):
        self.agents = []  # List to store agents
        for i in range(env.agent_number):  # Each agent runs DDPG
            self.agents.append(
                DDPG(state_dim[i], action_dim[i], critic_input_dim,
                     hidden_dim_a, hidden_dim_c, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):  # Return actor networks
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):  # Return target actor networks
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        ''' Take states of all agents, return list of action arrays for 4 agents '''
        states = [
            torch.tensor(np.array([states[i]]), dtype=torch.float, device=self.device)
            for i in range(env.agent_number)
        ]  # Read and format states
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        '''
        MADDPG update
        :param sample: Sampled experience containing:
                 ([obs of 4 agents], [actions of 4 agents], [rewards of 4 agents], [next obs of 4 agents])
        :param i_agent: Agent index
        '''
        obs, act, rew, next_obs = sample  # Unpack experience
        cur_agent = self.agents[i_agent]  # Get current agent

        # Update critic
        cur_agent.critic_optimizer.zero_grad()  # Clear gradients
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]  # Actions from target actor
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(target_critic_input)  # TD target
        critic_input = torch.cat((*obs, *act), dim=1)  # Centralized critic input
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())  # Loss between estimated and target
        critic_loss.backward()  # Backpropagate
        cur_agent.critic_optimizer.step()  # Update parameters

        # Update actor
        cur_agent.actor_optimizer.zero_grad()  # Clear gradients
        cur_actor_out = cur_agent.actor(obs[i_agent])  # Actor output
        cur_act_vf_in = gumbel_softmax(cur_actor_out)  # Convert to action
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # Regularization
        actor_loss.backward()  # Backpropagate
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        ''' Soft update target networks for all agents '''
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def update_D2LT(reward, agent_number, V_l1, V_l1_):
    '''
    Update access opportunity counters for each agent
    param reward: Transmission status of each agent
    param agent_number: Number of agents
    param V_l1: Previous A2LT for each agent (wi in paper)
    param V_l1_: Previous v(-i) for each agent (w(-i) in paper)
    '''
    V_l1 = [x + 1 for x in V_l1]  # Increment all counters
    V_l1_ = [x + 1 for x in V_l1_]
    for i in range(agent_number):
        if reward[i] == 1:  # Agent i successfully transmitted
            V_l1[i] = 0
            temp = V_l1_.copy()
            V_l1_ = np.zeros([agent_number, ])  # Reset others' v-i
            V_l1_[i] = temp[i]

    return V_l1, V_l1_


def normalize_D2LT(V_l1, V_l1_):
    '''
    Normalize counters to ci and c-i (paper notation)
    '''
    LEN = len(V_l1_)
    d_l1, d_l1_ = np.zeros([LEN, ]), np.zeros([LEN, ])  # Initialize Di and D-i
    for i in range(len(V_l1)):  # Normalize
        d_l1[i] = V_l1[i] / (V_l1[i] + V_l1_[i])
        d_l1_[i] = V_l1_[i] / (V_l1[i] + V_l1_[i])
    return d_l1, d_l1_


def revise_reward(reward, D, if_collision, alpha):
    '''
    Modify reward to include fairness: r_individual and r_global
    Weighted combination gives r_total
    '''
    reward_ind, reward_other = np.zeros([len(reward, )]), np.zeros([len(reward, )])
    reward_global_temp = 0

    # Calculate r_individual and r_global
    for i in range(len(reward)):
        reward_other[i] = sum(reward) - reward[i]
        if reward[i] == 1 and D[i] == np.max(D):  # Success & highest priority
            reward_global_temp = 1
            reward_ind[i] = 1

        if reward[i] == 1 and D[i] != np.max(D):  # Success but not highest priority
            reward_global_temp = D[i]
            reward_ind[i] = -1

        if reward[i] == 0 and D[i] == np.max(D):  # No transmission but should have
            reward_ind[i] = -1 / (1 - D[i])
        if reward[i] == 0 and D[i] != np.max(D):  # No transmission & shouldn't
            reward_ind[i] = 1

        if if_collision[i] == 1 and D[i] == np.max(D):  # Collision & highest priority
            reward_global_temp = -1
            reward_ind[i] = 1
        if if_collision[i] == 1 and D[i] != np.max(D):  # Collision & not highest
            reward_global_temp = -1
            reward_ind[i] = -1

    reward_global = np.array([reward_global_temp] * len(reward))

    # Weighted total reward
    r_total = alpha * reward_global + (1 - alpha) * reward_ind

    return reward_ind, reward_global, r_total, reward_other


########################################## Main Training Loop ############################################

num_episodes = 1
episode_length = 800000  # Max length per episode (interpreted as TXOPs)
buffer_size = 100000
hidden_dim_a = 64  # Actor hidden dimension
hidden_dim_c = 128  # Critic hidden dimension
actor_lr = 5e-4  # Learning rates
critic_lr = 5e-4
gamma = 0.95
tau = 1e-2  # Soft update parameter
batch_size = 64
device = torch.device("cpu")  # Use CPU for training
update_interval = 200
learning_interval = 100
minimal_size = 4000  # Minimum experience buffer size before learning
state_length_M = 5  # History length
alpha = 0.3  # Reward weighting factor
env = env_SL_4nd(state_length=state_length_M)  # Create environment
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

# State dimensions for each agent:
# 8 = action (2) + link observation (4) + ci + c(-i)
state_dims = [8 * state_length_M] * env.agent_number

# Action dimensions (one-hot)
action_dims = [2] * env.agent_number
critic_input_dim = sum(state_dims) + sum(action_dims)  # Total input size

# Initialize MADDPG
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim_a, hidden_dim_c, state_dims,
                action_dims, critic_input_dim, gamma, tau)

return_list = []  # To record returns
total_step = 0
time_stamp = 0  # Track simulation time

# Main training loop (single link scenario)
for i_episode in range(num_episodes):  # Typically one episode is sufficient
    state = env.reset()  # Initialize states
    # Track transmission results
    reward1_l1_list, reward2_l1_list, reward3_l1_list, reward4_l1_list = [], [], [], []
    V_l1 = np.zeros([env.agent_number, ])  # A2LT counters (wi)
    V_l1_ = np.zeros([env.agent_number, ])  # v(-i) counters
    d_l1 = np.zeros([env.agent_number, ])  # Normalized ci
    d_l1_ = np.zeros([env.agent_number, ])  # Normalized c(-i)
    D_l1 = np.zeros([env.agent_number, ])  # Priority weights Di
    next_state = state.copy()

    for e_i in tqdm(range(episode_length)):  # Each iteration = one access opportunity
        actions = maddpg.take_action(state, explore=True)  # Get actions from all agents

        ############# Environment Interaction ########
        link_obs, reward, l_collision = env.step(actions)  # Step environment
        # Normalize A2LT
        if e_i != 0:
            D_l1 = V_l1 / np.sum(V_l1)
        # Modify rewards
        r_ind, r_global, r_total, r_other = revise_reward(reward, D_l1, l_collision, alpha)
        #############################################

        # Update A2LT counters
        V_l1, V_l1_ = update_D2LT(reward, env.agent_number, V_l1, V_l1_)

        # Normalize to ci and c(-i)
        d_l1, d_l1_ = normalize_D2LT(V_l1, V_l1_)
        for i in range(env.agent_number):  # Update states (sliding window)
            next_state[i] = np.concatenate([state[i][8:], actions[i], link_obs[i], [d_l1[i], d_l1_[i]]])

        # Handle different transmission outcomes
        if np.sum(reward) == 1:  # Successful transmission
            time_stamp += 72  # Slots occupied by successful transmission
            # Record rewards
            reward1_l1_list += [reward[0]] * 72
            reward2_l1_list += [reward[1]] * 72
            reward3_l1_list += [reward[2]] * 72
            reward4_l1_list += [reward[3]] * 72

        elif np.sum(l_collision) > 0:  # Collision
            time_stamp += 70  # Slots occupied by collision
            reward1_l1_list += [reward[0]] * 70
            reward2_l1_list += [reward[1]] * 70
            reward3_l1_list += [reward[2]] * 70
            reward4_l1_list += [reward[3]] * 70

        else:  # Idle channel
            time_stamp += 1
            reward1_l1_list.append(reward[0])
            reward2_l1_list.append(reward[1])
            reward3_l1_list.append(reward[2])
            reward4_l1_list.append(reward[3])

        # Store experience
        replay_buffer.add(state, actions, r_total, next_state)

        # Transition to next state
        state = next_state.copy()
        total_step += 1

        # Learning phase
        if replay_buffer.size() >= minimal_size and total_step % learning_interval == 0:
            sample = replay_buffer.sample(batch_size)


            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x]
                              for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]


            sample = [stack_array(x) for x in sample]
            for a_i in range(env.agent_number):
                maddpg.update(sample, a_i)

            if total_step % update_interval == 0:  # Delayed target update
                maddpg.update_all_targets()

# Save reward data
with open(f'rewards/reward1_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
          f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt', 'w') as reward1_l1_txt:
    for i in reward1_l1_list:
        reward1_l1_txt.write(str(i) + '   ')
with open(f'rewards/reward2_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
          f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt', 'w') as reward2_l1_txt:
    for i in reward2_l1_list:
        reward2_l1_txt.write(str(i) + '   ')
with open(f'rewards/reward3_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
          f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt', 'w') as reward3_l1_txt:
    for i in reward3_l1_list:
        reward3_l1_txt.write(str(i) + '   ')
with open(f'rewards/reward4_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}_c_lr{critic_lr}'
          f'_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt', 'w') as reward4_l1_txt:
    for i in reward4_l1_list:
        reward4_l1_txt.write(str(i) + '   ')

# Save trained models
MODEL_PATH = ['model/4node_PSL_v2_1.pt', 'model/4node_PSL_v2_2.pt', 'model/4node_PSL_v2_3.pt', 'model/4node_PSL_v2_4.pt']
for i in range(env.agent_number):
    torch.save(maddpg.agents[i].actor.state_dict(), MODEL_PATH[i])
print("Total simulation time (slots):", time_stamp)
