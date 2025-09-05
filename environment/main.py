import numpy as np
import matplotlib.pyplot as plt
from warehouse_env import WarehouseEnv

env = WarehouseEnv(grid_size=(8, 8), num_shelf_types=3, num_shelves=3)

def encode_state(obs):
    # Get robot position
    layer_0 = obs["layers"][0]
    robot_y, robot_x = np.argwhere(layer_0 == 1)[0]

    # Get shelf positions and types
    shelf_layer = obs["layers"][1]
    # Get nearest shelf position and type
    shelf_positions = np.argwhere(shelf_layer > 0)
    nearest_shelf_dist = float("inf")
    nearest_shelf_type = -1

    for sy, sx in shelf_positions:
        if shelf_layer[sy, sx] < 6:  # Not a container
            dist = abs(robot_x - sx) + abs(robot_y - sy)
            if dist < nearest_shelf_dist:
                nearest_shelf_dist = dist
                nearest_shelf_type = int(shelf_layer[sy, sx])

    # Get container position and distance
    container_positions = np.argwhere(shelf_layer == 6)
    if len(container_positions) > 0:
        cy, cx = container_positions[0]
        container_dist = abs(robot_x - cx) + abs(robot_y - cy)
    else:
        container_dist = -1    

    # Inv and order info
    inv = tuple(obs["robot_inventory"].astype(int))
    order = tuple(obs["current_order"].astype(int))
    container = tuple(obs["container_contents"])

    state = (robot_x, robot_y, nearest_shelf_dist, nearest_shelf_type, 
             container_dist) + inv + order + container
    
    return state

# hyperparameters
alpha = 0.05
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9999
min_epsilon = 0.1
num_episodes = 10000 

display_interval = 1000

Q = {}
total_rewards = []
epsilon_values = []
pickups_per_episode = []
timeouts = []
completions = []
timeout_count = 0
completion_count = 0

obs = env.reset()
done = False

env.render()
# print(env.action_space)
# print(env.shelves)

for episode in range(num_episodes):
    pickups = 0
    drops = 0    
        
    observation, _ = env.reset() 

    env._reset_info()

    state = encode_state(observation)
    if state not in Q:
            Q[state] = np.zeros(env.action_space.n)
    done = False
    total_reward = 0           

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

    pickups += info["successful_pickup"]
    drops += info["successful_drop"]
    episode_reason = info["episode_reason"]

    if info["episode_reason"] == "timeout":
        timeout_count += 1
    elif info["episode_reason"] == "order_completed":
        completion_count += 1

    timeouts.append(timeout_count)        
    completions.append(completion_count)

    # End of episode
    total_rewards.append(total_reward)
    epsilon_values.append(epsilon)
    pickups_per_episode.append(pickups)

    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    if episode % display_interval == 0:
        # env.render()
        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}, Pickups: {pickups}, Drops: {drops}, Episode_reason: {episode_reason}")


plt.figure(figsize=(15, 10))

# Plot 1: Total Reward with Smoothed Trend
plt.subplot(2, 2, 1)
plt.plot(total_rewards, alpha=0.3, color='blue', label='Raw')
window_size = 100
if len(total_rewards) >= window_size:
    smoothed = [np.mean(total_rewards[max(0, i-window_size):i+1]) for i in range(len(total_rewards))]
    plt.plot(smoothed, linewidth=2, color='darkblue', label='Smoothed')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()

# Plot 2: Cumulative Completion Stats
plt.subplot(2, 2, 2)
plt.plot(completions, color='green', label='Orders Completed')
plt.plot(timeouts, color='red', label='Timeouts')
plt.xlabel('Episode')
plt.ylabel('Count')
plt.title('Cumulative Episode Outcomes')
plt.legend()

# Plot 3: Completion Rate (completed/(completed+timeouts))
plt.subplot(2, 2, 3)
completion_rates = []
for i in range(len(completions)):
    total = completions[i] + timeouts[i]
    rate = completions[i] / total if total > 0 else 0
    completion_rates.append(rate)
plt.plot(completion_rates, color='purple')
plt.xlabel('Episode')
plt.ylabel('Completion Rate')
plt.title('Order Completion Rate')
plt.ylim(0, 1)

# Plot 4: Pickups per Episode
plt.subplot(2, 2, 4)
plt.plot(pickups_per_episode)
plt.xlabel('Episode')
plt.ylabel('Pickups')
plt.title('Pickups per Episode')

plt.tight_layout()
plt.show()