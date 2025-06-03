import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm

from env_singlelink import env_SL_4nd  # Import environment from another py file


# Function definitions and training code remain identical
def onehot_from_logits(logits, eps=0.01):
    ''' Generate one-hot encoding of optimal action '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # Generate random actions and convert to one-hot
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # Use epsilon-greedy to select actions
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
    """
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)  # Uniform(0,1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Sample from Gumbel-Softmax distribution """
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)  # Add Gumbel noise to logits
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """ Sample from Gumbel-Softmax and discretize """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)  # Convert to one-hot
    y = (y_hard.to(logits.device) - y).detach() + y  # Preserve value with gradient
    # Return one-hot with gradients flowing through y
    return y


class TwoLayerFC(torch.nn.Module):
    ''' A 3-layer fully connected network '''

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
    ''' A 4-layer network with GRU structure '''

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
        state_dim, action_dim: For actor network (per-agent)
        critic_input_dim: Aggregated info from all agents (centralized critic)
        hidden_dim_a: 64 (actor)
        hidden_dim_c: 128 (critic, wider network)
        '''

        # Actor-critic + target networks (4 networks total)
        # Decentralized actor per agent
        self.actor = FCwithGRU(state_dim, action_dim, hidden_dim_a).to(device)
        self.target_actor = FCwithGRU(state_dim, action_dim,
                                      hidden_dim_a).to(device)

        # Centralized critic shared by all agents
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim_c).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim_c).to(device)

        # Synchronize target networks with main networks
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
        :return: Action as (4,) array for link choices
        '''
        action = self.actor(state)  # Get action from actor
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
        '''
        Multi-Agent DDPG implementation
        '''
        self.agents = []  # List of agents
        for i in range(env.agent_number):  # Each agent runs DDPG
            self.agents.append(
                DDPG(state_dim[i], action_dim[i], critic_input_dim,
                     hidden_dim_a, hidden_dim_c, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):  # Get actor networks
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):  # Get target actor networks
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        ''' Take actions for all agents given their states '''
        states = [
            torch.tensor(np.array([states[i]]), dtype=torch.float, device=self.device)
            for i in range(env.agent_number)
        ]  # Format states per agent
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        '''
        MADDPG update step
        :param sample: Sampled experience (obs, actions, rewards, next_obs)
        :param i_agent: Agent index to update
        '''
        obs, act, rew, next_obs = sample  # Unpack experience
        cur_agent = self.agents[i_agent]  # Get current agent

        # Critic update
        cur_agent.critic_optimizer.zero_grad()  # Reset gradients
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]  # Target actor actions
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(target_critic_input)  # TD target
        critic_input = torch.cat((*obs, *act), dim=1)  # Centralized critic input
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())  # MSE loss
        critic_loss.backward()  # Backpropagate
        cur_agent.critic_optimizer.step()  # Update critic

        # Actor update
        cur_agent.actor_optimizer.zero_grad()  # Reset gradients
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
        ''' Soft update all target networks '''
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def return_throughput(rewards, link_number):
    ''' Calculate throughput with smoothing '''
    N = int(len(rewards) / 10)
    print(N)
    temp_sum = 0
    throughput = []
    rate_list = [107.4, 202.5, 376.5]  # Ideal max rates for 2.4G/5G/6G
    rate = rate_list[link_number - 1]

    for i in range(len(rewards)):
        if i < N:
            temp_sum += rewards[i]
            throughput.append(temp_sum * rate / (i + 1))  # Long-term throughput
        else:
            temp_sum += rewards[i] - rewards[i - N]
            throughput.append(temp_sum * rate / N)  # Short-term throughput
    return throughput


def update_D2LT(reward, agent_number, V, V_):
    '''
    Update D2LT values (D2LT = Duration Since Last Transmission)
    :param reward: Transmission success flags per agent
    :param agent_number: Number of agents
    :param V: Current D2LT values
    :param V_: Current v(-i) values (D2LT of others)
    :return: Updated D2LT vectors
    '''
    V = [x + 1 for x in V]  # Increment all D2LT
    V_ = [x + 1 for x in V_]
    for i in range(agent_number):
        if reward[i] == 1:  # Agent i transmitted successfully
            V[i] = 0  # Reset its own D2LT
            temp = V_.copy()
            V_ = np.zeros([agent_number, ])  # Reset others' v(-i)
            V_[i] = temp[i]  # Keep its previous v(-i)

    return V, V_


def normalize_D2LT(V, V_):
    ''' Normalize D2LT values to get c_i and c_{-i} '''
    LEN = len(V_)
    d, d_ = np.zeros([LEN, ]), np.zeros([LEN, ])  # Initialize Di and D_{-i}
    for i in range(len(V)):
        d[i] = V[i] / (V[i] + V_[i])
        d_[i] = V_[i] / (V[i] + V_[i])
    return d, d_


########################################## Above code = training code (function definitions) ########################
########################################## Evaluation section (main logic below) ####################################

hidden_dim_a = 64  # Actor hidden dimension
hidden_dim_c = 128  # Critic hidden dimension
actor_lr = 5e-4  # Learning rates
critic_lr = 5e-4
gamma = 0.95  # Discount factor
tau = 1e-2  # Soft update parameter

device = torch.device("cpu")  # Force CPU usage

state_length_M = 5
env = env_SL_4nd(state_length=state_length_M)  # Create environment

state_dims = [8 * state_length_M, 8 * state_length_M, 8 * state_length_M, 8 * state_length_M]  # State dimensions
action_dims = [2, 2, 2, 2]  # Action dimensions
critic_input_dim = sum(state_dims) + sum(action_dims)  # Centralized input size

# Pre-trained model paths
MODEL_PATH = ['model/4node_PSL_V2_1.pt', 'model/4node_PSL_V2_2.pt',
              'model/4node_PSL_V2_3.pt', 'model/4node_PSL_V2_4.pt']

# Instantiate MADDPG and load trained models
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim_a, hidden_dim_c, state_dims,
                action_dims, critic_input_dim, gamma, tau)
for i in range(env.agent_number):
    maddpg.agents[i].actor.load_state_dict(torch.load(MODEL_PATH[i]))

n_episode = 1
episode_length = 333334

# Throughput tracking lists
l1_throughput1_list = []
l1_throughput2_list = []
l1_throughput3_list = []
l1_throughput4_list = []

l2_throughput1_list = []
l2_throughput2_list = []
l2_throughput3_list = []
l2_throughput4_list = []

l3_throughput1_list = []
l3_throughput2_list = []
l3_throughput3_list = []
l3_throughput4_list = []

# Main evaluation loop
for epoch in range(10):  # Repeat experiment for averaging
    # Initialize reward trackers
    reward1_l1, reward1_l2, reward1_l3 = [], [], []
    reward2_l1, reward2_l2, reward2_l3 = [], [], []
    reward3_l1, reward3_l2, reward3_l3 = [], [], []
    reward4_l1, reward4_l2, reward4_l3 = [], [], []

    # Initialize D2LT (Duration Since Last Transmission) trackers
    V_l1, V_l1_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])
    V_l2, V_l2_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])
    V_l3, V_l3_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])

    # Normalized D2LT trackers
    d_l1, d_l1_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])
    d_l2, d_l2_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])
    d_l3, d_l3_ = np.zeros([env.agent_number, ]), np.zeros([env.agent_number, ])

    # Transmission opportunity trackers (TXOP)
    TXOP1, TXOP2, TXOP3 = np.zeros([episode_length, ]), np.zeros([episode_length, ]), np.zeros([episode_length, ])

    for _ in range(n_episode):  # Episode loop
        # Initialize environment
        state1 = env.reset()
        next_state1 = state1.copy()
        state2 = env.reset()
        next_state2 = state2.copy()
        state3 = env.reset()
        next_state3 = state3.copy()

        for t_i in tqdm(range(episode_length)):  # Time slot loop
            # Link 1 processing
            if TXOP1[t_i] == 0:  # Check transmission opportunity
                actions_l1 = maddpg.take_action(state1, explore=False)
                link1_obs, reward_l1, l1_collision = env.step(actions_l1)

                # Update D2LT metrics
                V_l1, V_l1_ = update_D2LT(reward_l1, env.agent_number, V_l1, V_l1_)
                d_l1, d_l1_ = normalize_D2LT(V_l1, V_l1_)

                # Update state
                for i in range(env.agent_number):
                    next_state1[i] = np.concatenate([state1[i][8:], actions_l1[i], link1_obs[i], [d_l1[i], d_l1_[i]]])

                # Handle transmission results
                if np.sum(reward_l1) == 1:  # Successful transmission
                    TXOP1[t_i:t_i + 136] = 1  # Block next 136 slots
                    # Append rewards
                    reward1_l1 += [reward_l1[0]] * 136
                    reward2_l1 += [reward_l1[1]] * 136
                    reward3_l1 += [reward_l1[2]] * 136
                    reward4_l1 += [reward_l1[3]] * 136

                elif np.sum(l1_collision) > 0:  # Collision occurred
                    TXOP1[t_i:t_i + 134] = 1  # Block next 134 slots
                    reward1_l1 += [reward_l1[0]] * 134
                    reward2_l1 += [reward_l1[1]] * 134
                    reward3_l1 += [reward_l1[2]] * 134
                    reward4_l1 += [reward_l1[3]] * 134

                else:  # Idle channel
                    reward1_l1.append(0)
                    reward2_l1.append(0)
                    reward3_l1.append(0)
                    reward4_l1.append(0)
                state1 = next_state1.copy()

            # Link 2 processing (similar to link 1)
            if TXOP2[t_i] == 0:
                actions_l2 = maddpg.take_action(state2, explore=False)
                link2_obs, reward_l2, l2_collision = env.step(actions_l2)
                V_l2, V_l2_ = update_D2LT(reward_l2, env.agent_number, V_l2, V_l2_)
                d_l2, d_l2_ = normalize_D2LT(V_l2, V_l2_)
                for i in range(env.agent_number):
                    next_state2[i] = np.concatenate([state2[i][8:], actions_l2[i], link2_obs[i], [d_l2[i], d_l2_[i]]])
                if np.sum(reward_l2) == 1:
                    TXOP2[t_i:t_i + 72] = 1
                    reward1_l2 += [reward_l2[0]] * 72
                    reward2_l2 += [reward_l2[1]] * 72
                    reward3_l2 += [reward_l2[2]] * 72
                    reward4_l2 += [reward_l2[3]] * 72

                elif np.sum(l2_collision) > 0:
                    TXOP2[t_i:t_i + 70] = 1
                    reward1_l2 += [reward_l2[0]] * 70
                    reward2_l2 += [reward_l2[1]] * 70
                    reward3_l2 += [reward_l2[2]] * 70
                    reward4_l2 += [reward_l2[3]] * 70

                else:
                    reward1_l2.append(0)
                    reward2_l2.append(0)
                    reward3_l2.append(0)
                    reward4_l2.append(0)
                state2 = next_state2.copy()

            # Link 3 processing (similar to link 1)
            if TXOP3[t_i] == 0:
                actions_l3 = maddpg.take_action(state1, explore=False)
                link3_obs, reward_l3, l3_collision = env.step(actions_l3)
                V_l3, V_l3_ = update_D2LT(reward_l3, env.agent_number, V_l3, V_l3_)
                d_l3, d_l3_ = normalize_D2LT(V_l3, V_l3_)
                for i in range(env.agent_number):
                    next_state3[i] = np.concatenate([state3[i][8:], actions_l3[i], link3_obs[i], [d_l3[i], d_l3_[i]]])
                if np.sum(reward_l3) == 1:
                    TXOP3[t_i:t_i + 39] = 1
                    reward1_l3 += [reward_l3[0]] * 39
                    reward2_l3 += [reward_l3[1]] * 39
                    reward3_l3 += [reward_l3[2]] * 39
                    reward4_l3 += [reward_l3[3]] * 39

                elif np.sum(l3_collision) > 0:
                    TXOP3[t_i:t_i + 37] = 1
                    reward1_l3 += [reward_l3[0]] * 37
                    reward2_l3 += [reward_l3[1]] * 37
                    reward3_l3 += [reward_l3[2]] * 37
                    reward4_l3 += [reward_l3[3]] * 37

                else:
                    reward1_l3.append(0)
                    reward2_l3.append(0)
                    reward3_l3.append(0)
                    reward4_l3.append(0)
                state3 = next_state3.copy()

    # Calculate and store throughputs
    l1_throughput1_list += [return_throughput(reward1_l1[0:episode_length], 1)]
    l1_throughput2_list += [return_throughput(reward2_l1[0:episode_length], 1)]
    l1_throughput3_list += [return_throughput(reward3_l1[0:episode_length], 1)]
    l1_throughput4_list += [return_throughput(reward4_l1[0:episode_length], 1)]

    l2_throughput1_list += [return_throughput(reward1_l2[0:episode_length], 2)]
    l2_throughput2_list += [return_throughput(reward2_l2[0:episode_length], 2)]
    l2_throughput3_list += [return_throughput(reward3_l2[0:episode_length], 2)]
    l2_throughput4_list += [return_throughput(reward4_l2[0:episode_length], 2)]

    l3_throughput1_list += [return_throughput(reward1_l3[0:episode_length], 3)]
    l3_throughput2_list += [return_throughput(reward2_l3[0:episode_length], 3)]
    l3_throughput3_list += [return_throughput(reward3_l3[0:episode_length], 3)]
    l3_throughput4_list += [return_throughput(reward4_l3[0:episode_length], 3)]

# Average results across epochs
l1_throughput1 = np.mean(l1_throughput1_list, axis=0)
l1_throughput2 = np.mean(l1_throughput2_list, axis=0)
l1_throughput3 = np.mean(l1_throughput3_list, axis=0)
l1_throughput4 = np.mean(l1_throughput4_list, axis=0)

l2_throughput1 = np.mean(l2_throughput1_list, axis=0)
l2_throughput2 = np.mean(l2_throughput2_list, axis=0)
l2_throughput3 = np.mean(l2_throughput3_list, axis=0)
l2_throughput4 = np.mean(l2_throughput4_list, axis=0)

l3_throughput1 = np.mean(l3_throughput1_list, axis=0)
l3_throughput2 = np.mean(l3_throughput2_list, axis=0)
l3_throughput3 = np.mean(l3_throughput3_list, axis=0)
l3_throughput4 = np.mean(l3_throughput4_list, axis=0)

# Calculate total throughputs
l1_sum_throughput = [l1_throughput1[i] + l1_throughput2[i] + l1_throughput3[i] + l1_throughput4[i]
                     for i in range(episode_length)]
l2_sum_throughput = [l2_throughput1[i] + l2_throughput2[i] + l2_throughput3[i] + l2_throughput4[i]
                     for i in range(episode_length)]
l3_sum_throughput = [l3_throughput1[i] + l3_throughput2[i] + l3_throughput3[i] + l3_throughput4[i]
                     for i in range(episode_length)]

# Get stable throughput values (last 1000 steps)
mean1 = np.mean(round(l1_sum_throughput[-1000], 2))
mean2 = np.mean(round(l2_sum_throughput[-1000], 2))
mean3 = np.mean(round(l3_sum_throughput[-1000], 2))

# Print summary
print("Link 1:", np.mean(round(l1_throughput1[-1000], 2)), np.mean(round(l1_throughput2[-1000], 2)),
      np.mean(round(l1_throughput3[-1000], 2)), np.mean(round(l1_throughput4[-1000], 2)), mean1)
print("Link 2:", np.mean(round(l2_throughput1[-1000], 2)), np.mean(round(l2_throughput2[-1000], 2)),
      np.mean(round(l2_throughput3[-1000], 2)), np.mean(round(l2_throughput4[-1000], 2)), mean2)
print("Link 3:", np.mean(round(l3_throughput1[-1000], 2)), np.mean(round(l3_throughput2[-1000], 2)),
      np.mean(round(l3_throughput3[-1000], 2)), np.mean(round(l3_throughput4[-1000], 2)), mean3)

# Prepare time axis (convert slots to seconds)
normalized_x = np.linspace(0, episode_length * 9 / 1000000, len(l1_sum_throughput))

# Create throughput visualization
fig = plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(normalized_x, l1_sum_throughput, c='r', label='Total')
plt.plot(normalized_x, l1_throughput1, c='b', label='MLD 1')
plt.plot(normalized_x, l1_throughput2, c='cyan', label='MLD 2')
plt.plot(normalized_x, l1_throughput3, c='orange', label='MLD 3')
plt.plot(normalized_x, l1_throughput4, c='green', label='MLD 4')
plt.ylim((0, 120))
plt.xlim(0, episode_length * 9 / 1000000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Time(s)", fontsize=16)
plt.ylabel("Throughput(Mbps)", fontsize=16)
plt.title("Link 1 (Rate = 114.7 Mbps)", fontsize=16)
plt.text(episode_length * 9 / 1000000 * 0.8, mean1 * 0.85, f'Total = {mean1}',
         family='Times New Roman',
         fontsize=16,
         fontweight='bold',
         color='red')
plt.legend(loc='upper left', fontsize=13)

plt.subplot(3, 1, 2)  # Link 2 plot
plt.plot(normalized_x, l2_sum_throughput, c='r', label='Total')
plt.plot(normalized_x, l2_throughput1, c='b', label='MLD 1')
plt.plot(normalized_x, l2_throughput2, c='cyan', label='MLD 2')
plt.plot(normalized_x, l2_throughput3, c='orange', label='MLD 3')
plt.plot(normalized_x, l2_throughput4, c='green', label='MLD 4')
plt.ylim((0, 240))
plt.xlim(0, episode_length * 9 / 1000000)
plt.xticks(fontsize=16)
plt.yticks([0, 40, 80, 120, 160, 200, 240], fontsize=16)
plt.xlabel("Time(s)", fontsize=16)
plt.ylabel("Throughput(Mbps)", fontsize=16)
plt.title("Link 2 (Rate = 229.4 Mbps)", fontsize=16)
plt.text(episode_length * 9 / 1000000 * 0.8, mean2 * 0.85, f'Total = {mean2}',
         family='Times New Roman',
         fontsize=16,
         fontweight='bold',
         color='red')
plt.legend(loc='upper left', fontsize=13)

plt.subplot(3, 1, 3)  # Link 3 plot
plt.plot(normalized_x, l3_sum_throughput, c='r', label='Total')
plt.plot(normalized_x, l3_throughput1, c='b', label='MLD 1')
plt.plot(normalized_x, l3_throughput2, c='cyan', label='MLD 2')
plt.plot(normalized_x, l3_throughput3, c='orange', label='MLD 3')
plt.plot(normalized_x, l3_throughput4, c='green', label='MLD 4')
plt.ylim((0, 500))
plt.xlim(0, episode_length * 9 / 1000000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Time(s)", fontsize=16)
plt.ylabel("Throughput(Mbps)", fontsize=16)
plt.title("Link 3 (Rate = 480.4 Mbps)", fontsize=16)
plt.text(episode_length * 9 / 1000000 * 0.8, mean3 * 0.85, f'Total = {mean3}',
         family='Times New Roman',
         fontsize=16,
         fontweight='bold',
         color='red')
plt.legend(loc='upper left', fontsize=13)
plt.subplots_adjust(hspace=0.6)
plt.show()