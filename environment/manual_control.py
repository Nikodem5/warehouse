import numpy as np
from warehouse_env import WarehouseEnv

def display_state(env, total_reward):
    obs = env.get_observation()
    print("\nCurrent State:")
    print("Robot Position:", env.robot_pos)
    print("Robot Inventory:", obs["robot_inventory"])
    print("Container Contents:", obs["container_contents"])
    print("Current Order:", obs["current_order"])
    print("Possible Actions: 0=Up, 1=Down, 2=Left, 3=Right")
    for i in range(env.num_shelf_types):
        print(f"{4 + i}=Pickup type {i}")
    print(f"{4 + env.num_shelf_types}=Drop")
    print("Reward for Last Action:", env.last_reward)
    print("Total Reward:", total_reward)
    render_map(env)

def render_map(env):
    grid_display = [["." for _ in range(env.grid_size[0])] for _ in range(env.grid_size[1])]

    for (x, y), shelf in env.shelves.items():
        grid_display[y][x] = str(shelf["type"])

    cx, cy = env.container_pos
    grid_display[cy][cx] = "C"

    rx, ry = env.robot_pos
    grid_display[ry][rx] = "R"

    print("\nWarehouse grid:")
    for row in grid_display:
        print(" ".join(row))

def main():
    env = WarehouseEnv(grid_size=(5, 5), num_shelf_types=2, num_shelves=1)
    obs, _ = env.reset()
    env.last_reward = 0  # Initialize last reward
    total_reward = 0  # Initialize total reward

    while True:
        display_state(env, total_reward)
        try:
            action = int(input("Enter action (0-9): "))
            if action < 0 or action >= env.num_actions:
                print("Invalid action. Please enter a number between 0 and", env.num_actions - 1)
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        obs, reward, done, info = env.step(action)
        env.last_reward = reward  # Store the last reward
        total_reward += reward  # Update total reward

        if done:
            print("Episode finished. Total Reward:", total_reward)
            obs, _ = env.reset()
            total_reward = 0  # Reset total reward for the next episode

if __name__ == "__main__":
    main()