from gymnasium.envs.registration import register

# The id used by the LwH thesis code (sparse reward is the Config default,
# episode cap matches LwH's --max-episode-length 2000).
register(
    id='uav-v0',
    entry_point='gym_uav.envs:UavDenseEnv',
    max_episode_steps=2000
)

register(
    id='uav-v1',
    entry_point='gym_uav.envs:UavDenseEnv',
    max_episode_steps=100
)
