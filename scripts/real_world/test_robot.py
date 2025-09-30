from real_world_env import RealWorldEnv

env = RealWorldEnv()

env.step(3)  # 3: rotate right
env.step(2)  # 2: rotate left
env.step(1)  # 1: move forward
env.step(0)  # 0: no movement (stand still)

env.get_observation()  # {rgb: array, depth: array, instruction: str}
