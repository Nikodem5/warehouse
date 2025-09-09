import numpy as np
import matplotlib.pyplot as plt
from environment.warehouse_env import WarehouseEnv
from agents.q_learning_agent import QLearningAgent

env = WarehouseEnv(grid_size=(5, 5), num_shelf_types=2, num_shelves=2)
agent = QLearningAgent(env)

num_episodes = 1000
display_interval = 100

total_rewards = []
timeouts = []
completions = []
timeout_count = 0
completion_count = 0
action_counts = np.zeros(env.action_space.n)
action_history = []

for episode in range(num_episodes):    
    observation, _ = env.reset()
    env._reset_info()

    state = agent.encode_state(observation)
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        action_counts[action] += 1

        next_obs, reward, done, info = env.step(action)
        next_state = agent.encode_state(next_obs)

        agent.learn(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

    if episode % 100 == 99:
        action_distribution = action_counts / np.sum(action_counts)
        action_history.append(action_distribution.copy())
    
    episode_reason = info["episode_reason"]
    if info["episode_reason"] == "timeout":
        timeout_count += 1
    elif info["episode_reason"] == "order_completed":
        completion_count += 1

    timeouts.append(timeout_count)        
    completions.append(completion_count)    
    total_rewards.append(episode_reward)
    
    if episode % display_interval == 0:        
        print(f"Episode: {episode:4d} | Reward: {episode_reward:6.1f} | Orders completed: {completion_count:4d} | Timeouts: {timeout_count:4d}")

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

# Plot 4: Action distribution
plt.subplot(2, 2, 4)
if action_history:
    action_history = np.array(action_history)
    episodes = np.arange(99, num_episodes, 100)

    for i in range(env.action_space.n):
        if i < 4:
            label = ['Up', 'Down', 'Left', 'Right'][i]
        elif i < 4 + env.num_shelf_types:
            label = f'Pickup {i-4}'
        else:
            label = 'Drop'
        plt.plot(episodes, action_history[:, i], label=label)
    
    plt.xlabel('Episode')
    plt.ylabel('Action Frequency')
    plt.title('Action Distribution Over Time')
    plt.legend()

plt.tight_layout()
plt.show()        

# Save the trained agent
agent.save("saved_models/trained_q_agent.pt")
print("Agent saved to trained_q_agent.pt")