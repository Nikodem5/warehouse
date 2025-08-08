import numpy as np 

class WarehouseEnv:
    def __init__(self):
        # env
        self.action_space = [("move", "up"), ("move", "down"), ("move", "right"), ("move", "left"), ("pickup", "quant"), ("drop",)]
                 
        # step tracking
        self.step_count = 0
        self.max_steps = 200

        # robot
        self.robot_pos = (8,8)
        self.robot_inventory = [0, 0, 0, 0, 0]

        # order
        self.current_order = [np.random.randint(0, 10) for _ in range(5)]

        # container
        self.container_contents = [0, 0, 0, 0, 0]
        self.container_pos = (5, 8)

        # shelves
        self.shelves = {}
        for i in range(5):
            x = i * 2 + 1
            for y in [1, 3, 5]:
                self.shelves[(x,y)] = {"type": i+1, "count": 8}

        # layers
        layer_0 = np.zeros((10, 10), dtype=int)
        layer_1 = np.zeros((10, 10), dtype=int)
        layer_2 = np.zeros((10, 10), dtype=int)

        layer_0[self.robot_pos[0], self.robot_pos[1]] = 1
        for (x,y), shelf in self.shelves.items():
            layer_1[y][x] = shelf["type"]
            layer_2[y][x] = shelf["count"]

        layer_1[self.container_pos[0], self.container_pos[1]] = 6

        self.layers = [layer_0, layer_1, layer_2]

    def reset(self):
        # step tracking
        self.step_count = 0
        self.max_steps = 200

        # robot
        self.robot_pos = (8,8)
        self.robot_inventory = [0, 0, 0, 0, 0]

        # order
        self.current_order = [np.random.randint(0, 10) for _ in range(5)]

        # container
        self.container_contents = [0, 0, 0, 0, 0]
        self.container_pos = (5, 8)

        # shelves
        self.shelves = {}
        for i in range(5):
            x = i * 2 + 1
            for y in [1, 3, 5]:
                self.shelves[(x,y)] = {"type": i+1, "count": 8}

        # layers
        layer_0 = np.zeros((10, 10), dtype=int)
        layer_1 = np.zeros((10, 10), dtype=int)
        layer_2 = np.zeros((10, 10), dtype=int)

        layer_0[self.robot_pos[0], self.robot_pos[1]] = 1
        for (x,y), shelf in self.shelves.items():
            layer_1[y][x] = shelf["type"]
            layer_2[y][x] = shelf["count"]

        layer_1[self.container_pos[0], self.container_pos[1]] = 6

        self.layers = [layer_0, layer_1, layer_2]

        return self.get_observation()
    
    def get_observation(self):
        grid = np.stack(self.layers, axis=0) # [3, 10, 10] shape

        observation = {
            "grid": grid,
            "inventory": self.robot_inventory,
            "current_order": self.current_order,
            "container": self.container_contents,
            "progress": self.step_count / self.max_steps   
        }

        return observation

    def update_layers(self):
        # Reset layers
        self.layers[0].fill(0)
        self.layers[1].fill(0)
        self.layers[2].fill(0)

        # Update robot position
        x, y = self.robot_pos
        self.layers[0][y, x] = 1

        # Update shelves and container
        for (sx, sy), shelf in self.shelves.items():
            self.layers[1][sy, sx] = shelf["type"]
            self.layers[2][sy, sx] = shelf["count"]

        cx, cy = self.container_pos
        self.layers[1][cy, cx] = 6

    def print_layers(self):
        print("Layer 0 (Robot Position):")
        print(self.layers[0])
        print("\nLayer 1 (Shelves and Container):")
        print(self.layers[1])
        print("\nLayer 2 (Shelf Item Count):")
        print(self.layers[2])
        print("\n")

    def step(self, action):
        reward = 0
        self.step_count += 1
        x, y = self.robot_pos

        if action[0] == "move":
            if action[1] == "up":
                if y > 0 and (x, y-1) not in self.shelves:
                    y -= 1
                    reward += 5
                else:
                    reward -= 1                    

            elif action[1] == "down":
                if y < 8 and (x, y+1) not in self.shelves:
                    y += 1
                    reward += 5
                else:
                    reward -= 1                    

            elif action[1] == "right":
                if x < 8 and (x+1, y) not in self.shelves:
                    x += 1
                    reward += 5
                else:
                    reward -= 1                    

            elif action[1] == "left":
                if x > 0 and (x-1, y) not in self.shelves:
                    x -= 1
                    reward += 5
                else:
                    reward -= 1
            
            self.robot_pos = (x, y)

        elif action[0] == "pickup":
            quantity = action[1]
            
            if (x+1, y) in self.shelves:
                shelf = self.shelves[x+1, y]
                item_type = shelf["type"]

                # check if the robot can carry the requested quantity
                if self.robot_inventory[item_type] + quantity <= 10:
                    if shelf["count"] >= quantity:
                        # check if the pickup is beneficial for the current order
                        if self.robot_inventory[item_type] < self.current_order[item_type]:
                            self.robot_inventory[item_type] += quantity
                            shelf["count"] -= quantity
                            reward += 10
                        else:
                            reward -= 5
                    else:
                        # partial pickup
                        if self.robot_inventory[item_type] < self.current_order[item_type]:
                            self.robot_inventory[item_type] += shelf["count"]
                            shelf["count"] = 0
                            reward += 5
                        else:
                            reward -= 5
                else:
                    max_quantity = 10 - self.robot_inventory[item_type]
                    if shelf["count"] >= max_quantity:
                        if self.robot_inventory[item_type] < self.current_order[item_type]:
                            self.robot_inventory[item_type] += max_quantity
                            shelf["count"] -= max_quantity
                            reward += 5
                        else:
                            reward -= 5
                    else:
                        if self.robot_inventory[item_type] < self.current_order[item_type]:
                            self.robot_inventory[item_type] += shelf["count"]
                            shelf["count"] = 0
                            reward += 5
                        else:
                            reward -= 5
            else:
                reward -= 5 # penalty for picking from nothing 

        elif action[0] == "drop":
            if (x+1, y) == self.container_pos or (x-1, y) == self.container_pos or (x, y+1) == self.container_pos or (x, y-1) == self.container_pos:
                for i in range(len(self.robot_inventory)):
                    self.container_contents[i] += self.robot_inventory[i]
                    self.robot_inventory[i] = 0
            else:
                reward -= 5 # penalty for dropping to nothing

            if self.container_contents == self.current_order:
                reward += 100 # reward for complete order
                self.container_contents = [0, 0, 0, 0, 0]
                self.current_order = [np.random.randint(0, 10) for _ in range(5)]
            else:
                reward -= 5 # penalty for incorrect or incomplete order

        self.update_layers()
        self.print_layers()   

        return self.get_observation(), reward, self.step_count >= self.max_steps
    
    def set_robot_position(self, new_position):
        self.robot_pos = new_position