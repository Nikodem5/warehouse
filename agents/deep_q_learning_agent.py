from PIL.GimpGradientFile import EPSILON
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)        
        states, actions, rewards, next_states, dones = zip(*samples)

        states_array = np.array(states)
        next_states_array = np.array(next_states)

        return (torch.tensor(states_array),
                torch.tensor(actions),
                torch.tensor(rewards),
                torch.tensor(next_states_array),
                torch.tensor(dones))

class DoubleDQNAgent:
    def __init__(self, state_dim, num_item_types, replay_buffer_size=10000, batch_size=64, gamma=0.99, alpha=0.01, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1, target_update=100, hidden_dim=128):
        self.state_dim = state_dim
        self.num_item_types = num_item_types
        self.action_dim = 4 + num_item_types + 1

        self.policy_net = Network(state_dim, self.action_dim, hidden_dim)
        self.target_net = Network(state_dim, self.action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=alpha)
        self.memory = ReplayMemory(replay_buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim) 
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, float(done)))