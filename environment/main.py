import numpy as np
import matplotlib.pyplot as plt
from warehouse_env import WarehouseEnv

env = WarehouseEnv(grid_size=(5, 5), num_shelf_types=1, num_shelves=3)

def encode_state(obs):
    layer_0 = obs["layers"][0]
    robot_y, robot_x = np.argwhere(layer_0 == 1)[0]
    inv = tuple(obs["robot_inventory"].astype(int))
    order = tuple(obs["current_order"].astype(int))
    return (robot_x, robot_y) + inv + order 

# hyperparameters
alpha = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.1
num_episodes = 10000 

Q = {}
total_rewards = []
epsilon_values = []
pickups_per_episode = []
orders_completed_per_episode = []

obs = env.reset()
done = False

print(env.action_space)
print(env.shelves)

for episode in range(num_episodes):
    observation, _ = env.reset() 
    state = encode_state(observation)
    if state not in Q:
            Q[state] = np.zeros(env.action_space.n)
    done = False
    total_reward = 0
    pickups = 0
    orders_completed = 0
    drops = 0

    # Choose action based on epsilon-greedy epsilon policy
    while not done:
        if np.random.uniform(0.0, 1.0) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Take action and encode next state
        next_obs, reward, done, info = env.step(action)
        next_state = encode_state(next_obs)

        # Initialize Q-table
        if next_state not in Q:
            Q[next_state] = np.zeros(env.action_space.n)

        # Update Q-table
        Q[state][action] = Q[state][action] + alpha * ( reward + gamma * np.max(Q[next_state]) - Q[state][action] )

        # Move to next state
        state = next_state
        total_reward += reward

        pickups += info["successful_pickups"]
        drops += info["successful_drops"]
        orders_completed += info["orders_completed"]

    # End of episode
    total_rewards.append(total_reward)
    epsilon_values.append(epsilon)
    pickups_per_episode.append(pickups)
    orders_completed_per_episode.append(orders_completed)

    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    if episode % 50 == 0:
        # env.render()
        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}, Pickups: {pickups}, Drops: {drops}, Orders Completed: {orders_completed}")

# Plotting the total rewards and epsilon values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')

plt.subplot(1, 2, 2)
plt.plot(epsilon_values)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay over Episodes')

plt.tight_layout()
plt.show()

# print('Learned Policy:')
# for state, actions in Q.items():
#     best_action = np.argmax(actions)
#     print(f"State: {state}, Best Action: {best_action}")