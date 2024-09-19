import pathlib
from typing import Any, Dict, Optional, Tuple
from loguru import logger
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as GymHumanoidEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray

from envs.humanoid.reward_helpers import *

from constants import DEMOS_DICT

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.5,
    "lookat": np.array((0.25, 0.0, 1.25)),
    "elevation": -10.0,
    "azimuth": 180
}

class HumanoidEnvCustom(GymHumanoidEnv):
    def __init__(
        self,
        episode_length=240,
        reward_type="remain_standing",
        render_mode: str = "rgb_array",
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        healthy_z_range: Tuple[float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        camera_config: Optional[Dict[str, Any]] = DEFAULT_CAMERA_CONFIG,
        textured: bool = True,
        **kwargs,
    ):
        terminate_when_unhealthy = False
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            render_mode=render_mode,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )
        env_file_name = None
        if textured:
            env_file_name = "humanoid_textured.xml"
        else:
            env_file_name = "humanoid.xml"
        model_path = str(pathlib.Path(__file__).parent / env_file_name)
        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        self.episode_length = episode_length
        self.num_steps = 0
        self.stage = 0  # For the stage detector reward function
        if camera_config:
            self.camera_id = -1

        
        self.reward_fn = reward_original

        self._ref_joint_states = np.array([])

        if reward_type in DEMOS_DICT:
            self._load_reference_joint_states(DEMOS_DICT[reward_type])
        else:
            logger.info(f"Warning: {reward_type} is not in the DEMOS_DICT. No reference joint states loaded.")

        # Spawned the humanoid not so high
        self.init_qpos[2] = 1.3

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        xy_position_before = mass_center(self.model, self.data)

        obs, reward, terminated, truncated, info = super().step(action)

        reward, info = self.reward_fn(self.data, model=self.model, 
                                        dt=self.dt,
                                        num_steps=self.num_steps,
                                        curr_stage=self.stage,
                                        timestep=self.model.opt.timestep,
                                        xy_position_before=xy_position_before,
                                        ctrl_cost=self.control_cost(action),
                                        healthy_reward=self.healthy_reward,
                                        forward_reward_weight=self._forward_reward_weight,
                                        ref_joint_states=self._ref_joint_states)

        self.num_steps += 1
        self.stage = int(info.get("stage", 0))
        terminated = self.num_steps >= self.episode_length
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        self.stage = 0
        return super().reset(seed=seed, options=options)

    def get_obs(self):
        return self._get_obs()

    def _load_reference_joint_states(self, joint_state_fp_list):
        """
        Parameters:
            joint_state_fp_list (list): list of path to the saved reference joint state

        Effects:
            self._ref_joint_states gets updated
        """
        ref_joint_states_list = []
        for fp in joint_state_fp_list:
            ref_joint_states_list.append(np.load(fp))
        self._ref_joint_states = np.stack(ref_joint_states_list)

        # logger.debug(f"Updated self._ref_joint_states: {self._ref_joint_states.shape}\n{self._ref_joint_states}")

def reward_original(data, **kwargs):
    model = kwargs.get("model", None)
    xy_position_before = kwargs.get("xy_position_before", None)
    dt = kwargs.get("dt", None)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    healthy_reward = kwargs.get("healthy_reward", None)

    forward_reward_weight = kwargs.get("forward_reward_weight", None)

    xy_position_after = mass_center(model, data)

    xy_velocity = (xy_position_after - xy_position_before) / dt
    x_velocity, y_velocity = xy_velocity

    forward_reward = forward_reward_weight * x_velocity

    rewards = forward_reward + healthy_reward - ctrl_cost

    info = {
            "reward_linvel": f"{forward_reward:.2f}",
            "reward_quadctrl": f"{-ctrl_cost:.2f}",
            "reward_alive": f"{healthy_reward:.2f}",
            "x_position": f"{xy_position_after[0]:.2f}",
            "y_position": f"{xy_position_after[1]:.2f}",
            "distance_from_origin": f"{np.linalg.norm(xy_position_after, ord=2):.2f}",
            "x_velocity": f"{x_velocity:.2f}",
            "y_velocity": f"{y_velocity:.2f}",
            "forward_reward": f"{forward_reward:.2f}",
        }
    
    return rewards, info
