import gymnasium as gym
import numpy as np
# from gymnasium import spaces

class WarehouseEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(10) # 0-3 move up, down, left, right; 4-8 pickup type 0-4; 9 drop all  

        self.observation_space = gym.spaces.Dict({
            "layers": gym.spaces.Box(low=0, high=8, shape=(3, 10, 10), dtype=np.int32), # 3 layers; 1 - robot pos; 2 - shelfs pos; 3 - shelfs inv
            "robot_inventory": gym.spaces.Box(0, 10, (5,), dtype=np.int32),
            "current_order": gym.spaces.Box(0, 10, (5,), dtype=np.int32),
            "container_contents": gym.spaces.Box(0, 10, (5,), dtype=np.int32)
        })

        self.grid_size = (10, 10)

        self.robot_pos = (8, 8)
        self.robot_inventory = np.zeros(5, dtype=np.int32)
        self.shelves = {}
        for i in range(5):
            x = i * 2 + 1
            for y in [1, 3, 5]:
                self.shelves[(x,y)] = {"type": i+1, "count": 8}
        self.container = {(5, 8): [0, 0, 0, 0, 0]}
        self.container_pos = (5, 8)
        self.container_contents = self.container[self.container_pos]
        self.current_order = np.zeros(5, dtype=np.int32)

        self.step_count = 0
        self.max_steps = 500
        self.replenish_interval = 10
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.robot_pos = (8, 8)
        self.robot_inventory = np.zeros(5, dtype=np.int32)
        self.shelves = {}
        for i in range(5):
            x = i * 2 + 1
            for y in [1, 3, 5]:
                self.shelves[(x,y)] = {"type": i+1, "count": 8}
        self.container = {(5, 8): [0, 0, 0, 0, 0]}
        self.current_order = self.np_random.integers(0, 11, size=5, dtype=np.int32)

        self.step_count = 0

        obs = self.get_observation()
        
        return obs, {}
    
    def get_observation(self):
        layers = np.zeros((3, 10, 10), dtype=np.int32)
        
        x, y = self.robot_pos
        layers[0, y, x] = 1

        for (x, y), shelf in self.shelves.items():
            layers[1, y, x] = shelf["type"] 

        for (x, y) in self.container.keys():
            layers[1, y, x] = 6

        for (x, y), shelf in self.shelves.items():
            layers[2, y, x] = shelf["count"]

        observation = {
            "layers": layers,
            "robot_inventory": self.robot_inventory.copy(),
            "current_order": self.current_order.copy(),
            "container_contents": list(self.container.values())[0].copy()
        }

        return observation
        
    def step(self, action):
        reward = 0
        done = False
        info = {}
        self.step_count += 1

        x, y = self.robot_pos

        # Movement actions 0-3    
        if action in [0, 1, 2, 3]: 
            new_x, new_y = self._compute_new_pos(action)
            if self._is_within_bounds(new_x, new_y):
                if (new_x, new_y) not in self.shelves and (new_x, new_y) != self.container_pos:
                    self.robot_pos = (new_x, new_y)
                    reward -= 1 # small step penalty
                else:
                    reward -= 5 # walked into a shelf
            else:
                reward -= 5 # walked out of bands

        # Pickup actions 4-8                
        elif action in [4, 5, 6, 7, 8]:
            item_type = action - 4
            shelf_pos = (x+1, y) 

            if shelf_pos in self.shelves:
                shelf = self.shelves[(x+1, y)]

                if shelf["type"] == item_type:
                    if shelf["count"] >= 1:
                        if sum(self.robot_inventory) < 10:
                            self.robot_inventory[item_type] += 1
                            shelf["count"] -= 1
                            if self.robot_inventory[item_type] <= self.current_order[item_type]:
                                reward += 15 # for beneficial pickup
                            else:
                                reward -= 5 # for useless pickup
                        else:
                            reward -= 10 # full inventory
                    else:
                        reward -= 5 # empty shelf
                else:
                    reward -= 2 # wrong item type
            else:
                reward -= 5 # no shelf to the right

        # Drop action 9                
        elif action == 9:
            cx, cy = self.container_pos
            adjacent_to_container = (
            (x + 1, y) == (cx, cy) or (x - 1, y) == (cx, cy) or
            (x, y + 1) == (cx, cy) or (x, y - 1) == (cx, cy)
            )
            
            if adjacent_to_container:
                for i in range(5):
                    self.container_contents[i] += self.robot_inventory[i]
                    self.robot_inventory[i] = 0

                # check if order is fulfilled
                if np.array_equal(self.container_contents, self.current_order):
                    reward += 1000 # for correct order
                    self.current_order = self.np_random.integers(0, 11, size=5, dtype=np.int32)
                    self.container_contents = np.zeros(5, dtype=np.int32)
                elif all(self.container_contents[i] <= self.current_order[i] for i in range(5)):
                    reward -= 10 # incomplete
                else:
                    reward -= 50 # wrong items
            else:
                reward -= 5 # drop not near container

        # Shelf replenish
        if self.step_count % self.replenish_interval == 0:
            for shelf in self.shelves.values():
                shelf["count"] = min(shelf["count"] + 2, 8)

        # Done check
        if self.step_count >= self.max_steps:
            done = True

        # Return
        observation = self.get_observation()
        return observation, reward, done, info


    def _compute_new_pos(self, action):
        dxdy = [(0, -1), (0, 1), (-1, 0), (1, 0)] # up, down, left, right
        dx, dy = dxdy[action]
        x, y = self.robot_pos
        return x + dx, y + dy
    
    def _is_within_bounds(self, x, y):
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
    
    def render(self):
        grid_display = [["." for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]

        for (x, y), shelf in self.shelves.items():
            grid_display[y][x] = str(shelf["type"])

        cx, cy = self.container_pos
        grid_display[cy][cx] = "C"

        rx, ry = self.robot_pos
        grid_display[ry][rx] = "R"

        print("Warehouse grid:")
        for row in grid_display:
            print(" ".join(row))

        print("\nRobot Inventory:", self.robot_inventory)
        print("Current Order:   ", self.current_order)
        print("Container:       ", self.container_contents)
        print("Step Count:      ", self.step_count)