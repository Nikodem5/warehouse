import numpy as np 

class WarehouseEnv:
    def __init__(self):
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
        pass

    def step(self):
        pass