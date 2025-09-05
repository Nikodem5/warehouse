import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, env, alpha=0.05, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = {}

    def encode_state(self, obs):
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

    def select_action(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.env.action_space.n)

        if np.random.uniform(0.0, 1.0) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.env.action_space.n)

        self.Q[state][action] = self.Q[state][action] + self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        )

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def save(self, path):        
        with open(path, 'wb') as f:
            pickle.dump({
                'Q': self.Q,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'gamma': self.gamma
            }, f)

    def load(self, path):    
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.Q = checkpoint['Q']
        self.epsilon = checkpoint['epsilon']
        self.alpha = checkpoint['alpha']
        self.gamma = checkpoint['gamma']
