from marlenv.marlenv.wrappers import make_snake, RenderGUI

env, obs_shape, action_shape, properties = make_snake(
    num_envs=1,
    num_snakes=4,
    height=20,
    width=20,
    snake_length=5,
    vision_range=5
)

env = RenderGUI(env)  # bọc thêm render GUI

obs = env.reset()
done = [False] * properties['num_snakes']

import time
while not all(done):
    env.render()
    actions = [env.action_space.sample() for _ in range(properties['num_snakes'])]
    obs, rewards, done, infos = env.step(actions)
    print("obs = ", obs)
    time.sleep(0.2)

env.close()
