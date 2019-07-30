import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='simple_mag_lev-v0',
    entry_point='gym_simple_mag_lev.envs:MagLevEnv',
)
