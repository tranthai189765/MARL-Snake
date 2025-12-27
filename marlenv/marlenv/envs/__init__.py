# Use Gymnasium's register when available to ensure registration goes into the same registry
try:
    from gymnasium.envs.registration import register
except Exception:
    from gym.envs.registration import register

register(
    id='Snake-v1',
    entry_point='marlenv.envs.snake_env:SnakeEnv',
)

register(
    id='SnakeGraph-v1',
    entry_point='marlenv.envs.graph_snake_env:GraphSnakeEnv',
)

register(
    id='SnakeCoop-v1',
    entry_point='marlenv.envs.coop_snake_env:CoopSnakeEnv'
)