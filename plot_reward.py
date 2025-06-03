import numpy as np
import matplotlib.pyplot as plt

# Plot training results - parameters must match training configuration
episode_length = 800000
update_interval = 200
actor_lr = 5e-4
critic_lr = 5e-4
gamma = 0.95
batch_size = 64
state_length_M = 5
alpha = 0.3
learning_interval = 100


def return_throughput(rewards):
    '''
    Performance metric represented as throughput:
    average number of successfully transmitted packets per time slot over N slots
    :param rewards: array of length 10000
    :return: throughput array of length 10000
    '''
    # N = int(len(rewards)/50)  # Original calculation
    N = 500000  # Window size for moving average
    temp_sum = 0
    throughput = []
    for i in range(len(rewards)):
        if i < N:
            temp_sum += rewards[i]
            # For initial window: average = sum/(i+1) (Long-Term Throughput)
            throughput.append(temp_sum * 229.4 / (i + 1))
        else:
            temp_sum += rewards[i] - rewards[i - N]
            # For full window: average = sum/N (Short-Term Throughput)
            throughput.append(temp_sum * 229.4 / N)
    return throughput


for idx in range(1, 2):
    # Load reward data for each agent from files
    reward1_l1 = np.loadtxt(f'rewards/reward1_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
                            f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt')
    reward2_l1 = np.loadtxt(f'rewards/reward2_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
                            f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt')
    reward3_l1 = np.loadtxt(f'rewards/reward3_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
                            f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt')
    reward4_l1 = np.loadtxt(f'rewards/reward4_l1_Len{episode_length}_UpdateInterval{update_interval}_a_lr{actor_lr}'
                            f'_c_lr{critic_lr}_gamma{gamma}_B{batch_size}_M{state_length_M}_α{alpha}_LL{learning_interval}_4mld_v2.txt')

    print(len(reward1_l1))
    slots = 11111111  # Number of time slots to visualize
    print(slots)

    # Calculate throughput for each agent on link 1
    l1_throughput1 = return_throughput(reward1_l1[0:slots])  # Agent1 throughput
    l1_throughput2 = return_throughput(reward2_l1[0:slots])  # Agent2 throughput
    l1_throughput3 = return_throughput(reward3_l1[0:slots])  # Agent3 throughput
    l1_throughput4 = return_throughput(reward4_l1[0:slots])  # Agent4 throughput

    # Calculate total throughput for link 1
    l1_sum_throughput = [
        l1_throughput1[i] + l1_throughput2[i] + l1_throughput3[i] + l1_throughput4[i]
        for i in range(slots)
    ]

    # Calculate mean of last 1000 samples for convergence value
    mean1 = np.mean(round(l1_sum_throughput[-1000], 2))

    # Normalize x-axis to seconds
    normalized_x = np.linspace(0, slots * 9 / 1000000, slots)

    # Create figure with size ratio 14:6
    fig = plt.figure(figsize=(14, 7))
    plt.plot(normalized_x, l1_sum_throughput, c='r', label='Total')
    plt.plot(normalized_x, l1_throughput1[0:slots], c='b', label='MLD 1')
    plt.plot(normalized_x, l1_throughput2[0:slots], c='cyan', label='MLD 2')
    plt.plot(normalized_x, l1_throughput3[0:slots], c='orange', label='MLD 3')
    plt.plot(normalized_x, l1_throughput4[0:slots], c='green', label='MLD 4')

    # Set plot limits and labels
    plt.ylim(0, 250)
    plt.xlim(0, 100)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Time(s)", fontsize=18)
    plt.ylabel("Throughput(Mbps)", fontsize=18)

    # Add convergence value annotation
    plt.text(80, mean1 * 0.9, f'Total = {mean1}',
             family='Times New Roman',
             fontsize=18,
             fontweight='bold',
             color='red'
             )
    plt.legend(fontsize=18)

    plt.show()