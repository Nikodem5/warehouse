import gymnasium as gym
import numpy as np

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), num_shelf_types=2, num_shelves=2):
        # Constants
        self.MAX_INVENTORY = 10
        self.MAX_SHELF_CAPACITY = 8
        self.REPLENISH_AMOUNT = 2
        self.MAX_STEPS = 600
        self.REPLENISH_INTERVAL = 10
        self.MIN_ORDER_SIZE = 1
        self.MAX_ORDER_SIZE = 5
        
        self.grid_size = grid_size
        self.num_shelf_types = num_shelf_types
        self.num_shelves = num_shelves
        self.info = {
            "successful_pickup": 0,
            "episode_reason": "",
            "successful_drop": 0            
        }
        self.done = False
        self.order_success = False
        
        self.num_actions = 4 + self.num_shelf_types + 1 # 4 movement + pickups + drop
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Dict({
            "layers": gym.spaces.Box(low=0, high=8, shape=(3, *self.grid_size), dtype=np.int32),
            "robot_inventory": gym.spaces.Box(0, self.MAX_INVENTORY, (self.num_shelf_types,), dtype=np.int32),
            "current_order": gym.spaces.Box(0, self.MAX_INVENTORY, (self.num_shelf_types,), dtype=np.int32),
            "container_contents": gym.spaces.Box(0, self.MAX_INVENTORY, (self.num_shelf_types,), dtype=np.int32)
        })

        self.robot_pos = (self.grid_size[0] - 2, self.grid_size[1] - 2)
        self.robot_inventory = np.zeros(self.num_shelf_types, dtype=np.int32)
        self.shelves = self._create_shelves()
        self.container_pos = (self.grid_size[0] // 2, self.grid_size[1] - 2)
        self.container = {self.container_pos: [0] * self.num_shelf_types}        
        self.container_contents = self.container[self.container_pos]
        self.current_order = np.zeros(self.num_shelf_types, dtype=np.int32)
        self.step_count = 0
        self.np_random = np.random.default_rng()                        

    def _create_shelves(self):
        shelves = {}
        for i in range(self.num_shelf_types):
            x = (i * 2 + 1) % self.grid_size[0]
            for j in range(self.num_shelves):
                y = (j * 2 + 1) % self.grid_size[1]
                shelves[(x, y)] = {"type": i, "count": self.MAX_SHELF_CAPACITY}

        return shelves
    
    def _generate_new_order(self):
        return self.np_random.integers(self.MIN_ORDER_SIZE, self.MAX_ORDER_SIZE, size=self.num_shelf_types, dtype=np.int32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.done = False
        self.robot_pos = (self.grid_size[0] - 2, self.grid_size[1] - 2)
        self.robot_inventory = np.zeros(self.num_shelf_types, dtype=np.int32)
        self.shelves = self._create_shelves()
        self.container = {self.container_pos: [0] * self.num_shelf_types} 
        self.container_contents = self.container[self.container_pos]
        self.current_order = self._generate_new_order()
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
            "container_contents": self.container[self.container_pos].copy()
        }

        return observation
        
    def step(self, action):
        reward = 0
        self.step_count += 1
        self.done = False      
        self.order_success = False

        # Handle actions
        if action in [0, 1, 2, 3]:
            reward += self._handle_movement(action)
        elif 4 <= action < 4 + self.num_shelf_types:
            reward += self._handle_pickup(action)
        elif action == 4 + self.num_shelf_types:
            reward += self._handle_drop()

        # Replenish shelves
        self._replenish_shelves()

        # Check if done
        self.done = self._check_done()      

        # Get observation
        observation = self.get_observation()

        return observation, reward, self.done, self.info

    def _handle_movement(self, action):
        reward = -0.1
        x, y = self.robot_pos
        new_x, new_y = self._compute_new_pos(action)

        if self._is_within_bounds(new_x, new_y):
            if (new_x, new_y) not in self.shelves and (new_x, new_y) != self.container_pos:
                self.robot_pos = (new_x, new_y)                
            else:
                reward -= 0.5 # walked into a shelf or a container
        else:
            reward -= 0.5 # walked out of bounds

        return reward    

    def _handle_pickup(self, action):
        reward = 0
        item_type = action - 4
        x, y = self.robot_pos
        adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        successful_pickup = False

        for shelf_pos in adjacent_positions:
            if shelf_pos in self.shelves:
                shelf = self.shelves[shelf_pos]

                if shelf["type"] == item_type:
                    if shelf["count"] >= 1:
                        if sum(self.robot_inventory) < 10:
                            reward += 1 # small reward for successful pickup
                            self.info["successful_pickup"] += 1
                            self.robot_inventory[item_type] += 1
                            shelf["count"] -= 1
                            successful_pickup = True

                            if self.robot_inventory[item_type] <=self.current_order[item_type]:
                                reward += 10 # for beneficial pickup                                
                            else:
                                reward -= 2 # for useless pickup                                
                            break # stop after successful pickup
                        else:
                            reward -= 5 # for full inventory
                            break # stop checking shelves if full inv                            
                    else:
                        reward -= 2 # for picking from an empty shelf
                        # no break continue looking for non empty shelf                        
                else:
                    reward -= 1 # picking wrong item type
                    # no break continue looking for correct type of shelf                    

        if not successful_pickup:
            reward -= 5 # no shelf nearby            

        return reward                

    def _handle_drop(self):
        reward = 0
        x, y = self.robot_pos
        cx, cy = self.container_pos
        adjacent_to_container = (
            (x + 1, y) == (cx, cy) or (x - 1, y) == (cx, cy) or
            (x, y + 1) == (cx, cy) or (x, y - 1) == (cx, cy)
        )

        if adjacent_to_container:
            items_dropped = 0
            # Drop everything into container
            for i in range(self.num_shelf_types):
                items_to_drop = min(self.robot_inventory[i], self.current_order[i] - self.container_contents[i])
                if items_to_drop > 0:
                    reward += 5 * items_to_drop # Progressive reward                    
                    self.container[self.container_pos][i] += items_to_drop
                    self.robot_inventory[i] -= items_to_drop
                    items_dropped += items_to_drop
                    self.info["successful_drop"] += 1
          

            # Check if order is fulfilled
            if all(self.container[self.container_pos][i] >= self.current_order[i] for i in range(self.num_shelf_types)):
                reward += 100
                self.info["episode_reason"] = "order_completed"                
                self.order_success = True
                # self.current_order = self.np_random.integers(1, 5, size=self.num_shelf_types, dtype=np.int32)
                self.container[self.container_pos] = [0] * self.num_shelf_types
        else:
            reward -= 1 # not adjucent to container

        return reward    


    def _replenish_shelves(self):
        if self.step_count % self.REPLENISH_INTERVAL == 0:
            for shelf in self.shelves.values():
                shelf["count"] = min(shelf["count"] + self.REPLENISH_AMOUNT, 8)

    def _check_done(self):
        if self.step_count >= self.MAX_STEPS:
            self.info["episode_reason"] = "timeout"
            return True
        elif self.order_success:
            return True
        else:
            return False

    def _reset_info(self):
        self.info = {
        "successful_pickup": 0,
        "episode_reason": "",
        "successful_drop": 0        
    }

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