from real_world_env import RealWorldEnv
from stream import run, set_env

env = RealWorldEnv()

env.step(3)  # 3: rotate right
env.step(2)  # 2: rotate left
env.step(1)  # 1: move forward
env.step(0)  # 0: no movement (stand still)

obs = env.get_observation()  # {rgb: array, depth: array, instruction: str}
obs["instruction"] = "red"
print(obs["instruction"])
# meta = save_obs(obs, outdir="./captures1", prefix="test")
# print("Saved observation metadata:", meta)

# test stream

set_env(env)
print("--- start running steam app ---")
run()
