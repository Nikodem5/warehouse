import gymnasium as gym
import numpy as np
# from gymnasium import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), num_shelf_types=5, num_shelves=5):
        self.grid_size = grid_size
        self.num_shelf_types = num_shelf_types
        self.num_shelves = num_shelves
        
        self.num_actions = 4 + self.num_shelf_types + 1 # 4 movement + pickups + drop
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Dict({
            "layers": gym.spaces.Box(low=0, high=8, shape=(3, *self.grid_size), dtype=np.int32),
            "robot_inventory": gym.spaces.Box(0, 10, (self.num_shelf_types,), dtype=np.int32),
            "current_order": gym.spaces.Box(0, 10, (self.num_shelf_types,), dtype=np.int32),
            "container_contents": gym.spaces.Box(0, 10, (self.num_shelf_types,), dtype=np.int32)
        })

        self.robot_pos = (self.grid_size[0] - 2, self.grid_size[1] - 2)
        self.robot_inventory = np.zeros(self.num_shelf_types, dtype=np.int32)
        self.shelves = {}
        
        for i in range(self.num_shelf_types):
            x = (i * 2 + 1) % self.grid_size[0]
            for j in range(self.num_shelves):
                y = (j * 2 + 1) % self.grid_size[1] 
                self.shelves[(x,y)] = {"type": i, "count": 8}
        
        self.container_pos = (self.grid_size[0] // 2, self.grid_size[1] - 2)
        self.container = {self.container_pos: [0] * self.num_shelf_types}        
        self.container_contents = self.container[self.container_pos]

        self.current_order = np.zeros(self.num_shelf_types, dtype=np.int32)

        self.step_count = 0
        self.max_steps = 500
        self.replenish_interval = 10
        self.np_random = np.random.default_rng()
        self.holding_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.robot_pos = (self.grid_size[0] - 2, self.grid_size[1] - 2)
        self.robot_inventory = np.zeros(self.num_shelf_types, dtype=np.int32)
        self.shelves = {}

        for i in range(self.num_shelf_types):
            x = (i * 2 + 1) % self.grid_size[0]
            for j in range(self.num_shelves):
                y = (j * 2 + 1) % self.grid_size[1]
                self.shelves[(x, y)] = {"type": i, "count": 8}
        self.container = {self.container_pos: [0] * self.num_shelf_types} 
        # self.current_order = np.zeros(self.num_shelf_types, dtype=np.int32)
        self.current_order = self.np_random.integers(1, 11, size=self.num_shelf_types, dtype=np.int32)


        self.step_count = 0

        obs = self.get_observation()
        
        return obs, {}
    
    def get_observation(self):
        layers = np.zeros((3, *self.grid_size), dtype=np.int32)
        
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

        if sum(self.robot_inventory) > 0:
            self.holding_steps += 1
        else:
            self.holding_steps = 0

        if self.holding_steps >= 5:
            reward -= 0.1

        done = False
        info = {"successful_pickups": 0, "orders_completed": 0, "successful_drops": 0}
        self.step_count += 1

        x, y = self.robot_pos

        # Log currect state and action        
        # print(f"Step: {self.step_count}, Action: {action}, Position: {self.robot_pos}, Inventory: {self.robot_inventory}")

        # Movement actions 0-3    
        if action in [0, 1, 2, 3]: 
            new_x, new_y = self._compute_new_pos(action)
            if self._is_within_bounds(new_x, new_y):
                if (new_x, new_y) not in self.shelves and (new_x, new_y) != self.container_pos:
                    self.robot_pos = (new_x, new_y)
                    reward -= 0.5 # small step penalty
                else:
                    reward -= 1 # walked into a shelf
            else:
                reward -= 2 # walked out of bounds

        # Pickup actions 4-num_shelf_types                
        elif 4 <= action < 4 + self.num_shelf_types:
            item_type = action - 4            
            adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

            for shelf_pos in adjacent_positions:
                if shelf_pos in self.shelves:                    
                    shelf = self.shelves[shelf_pos]
                    
                    if shelf["type"] == item_type:              

                        if shelf["count"] >= 1:

                            if sum(self.robot_inventory) < 10:

                                reward += 1 # for successful pickup
                                self.robot_inventory[item_type] += 1
                                shelf["count"] -= 1
                                info["successful_pickups"] += 1

                                if self.robot_inventory[item_type] <= self.current_order[item_type]:
                                    reward += 10 # for beneficial pickup
                                else:
                                    reward -= 2 # for useless pickup
                            else:
                                reward -= 10 # full inventory
                        else:
                            reward -= 2 # empty shelf
                    else:
                        reward -= 1 # wrong item type
                else:
                    reward -= 5 # no shelf to the right

        # Drop action num_shelf_types + 1                
        elif action == 4 + self.num_shelf_types:
            cx, cy = self.container_pos
            adjacent_to_container = (
            (x + 1, y) == (cx, cy) or (x - 1, y) == (cx, cy) or
            (x, y + 1) == (cx, cy) or (x, y - 1) == (cx, cy)
            )
            
            if adjacent_to_container:
                for i in range(self.num_shelf_types):
                    self.container_contents[i] += self.robot_inventory[i]
                    self.robot_inventory[i] = 0
                    info["successful_drops"] += 1
                    reward += 20 * min(self.robot_inventory[i], self.current_order[i] - self.container_contents[i])
                    
                # check if order is fulfilled
                if np.array_equal(self.container_contents, self.current_order):
                    reward += 1000 # for correct order
                    print(f"order completed!!!!! order: {self.current_order}")
                    info["orders_completed"] += 1
                    self.current_order = self.np_random.integers(1, 11, size=self.num_shelf_types, dtype=np.int32)
                    self.container_contents = np.zeros(self.num_shelf_types, dtype=np.int32)
                elif all(self.container_contents[i] <= self.current_order[i] for i in range(self.num_shelf_types)):
                    reward -= 0.5 # incomplete
                else:
                    reward -= 5 # wrong items
            else:
                reward -= 10 # drop not near container

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