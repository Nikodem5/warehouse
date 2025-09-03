from warehouse_env import WarehouseEnv

env = WarehouseEnv()

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print("reward: ", reward)
    input("Press enter to continue...")