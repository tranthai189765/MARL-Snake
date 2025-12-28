from marlenv.marlenv.wrappers import make_snake, RenderGUI

env, obs_shape, action_shape, properties = make_snake(
    num_envs=1,
    num_snakes=4,
    height=20,
    width=20,
    snake_length=5,
    vision_range=5
)

env = RenderGUI(env)  # bật lại render GUI

obs = env.reset()
dones = [False] * properties['num_snakes']

import time
while not all(dones):
    env.render()
    actions = [env.action_space.sample() for _ in range(properties['num_snakes'])]
    obs, rewards, dones, infos = env.step(actions)
    print("Rewards:", rewards, "Dones:", dones)
    time.sleep(0.2)

env.close()