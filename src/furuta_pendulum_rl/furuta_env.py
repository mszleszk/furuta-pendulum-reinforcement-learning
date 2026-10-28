import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from furuta_pendulum_rl import constants

class FurutaEnv(gym.Env):
    def __init__(self, model_path=constants.PENDULUM_MODEL_PATH, frame_skip=5, max_episode_steps=1000):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self.action_space = spaces.Box(
            self.model.actuator_ctrlrange[:, 0].astype(np.float32), 
            self.model.actuator_ctrlrange[:, 1].astype(np.float32),
        )

        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(-np.pi, np.pi, (obs_dim,))

    def _get_obs(self):
        normalized_qpos = self._normalize_angle(self.data.qpos)
        return np.concatenate([normalized_qpos, self.data.qvel]).astype(np.float32)

    #[-pi, pi]
    def _normalize_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        base_angle = self.data.qpos[0]
        pendulum_angle = self.data.qpos[1]
        
        angle_diff = self._normalize_angle(pendulum_angle - np.pi)

        angle_reward = np.exp(-2.0 * (angle_diff ** 2))
        base_penalty = 0.001 * (base_angle ** 2)
        action_penalty = 0.0005 * np.sum(np.square(action))
        velocity_penalty = 0.001 * np.sum(np.square(self.data.qvel))

        reward = angle_reward - base_penalty - action_penalty - velocity_penalty

        terminated = bool(np.abs(angle_diff) > 1.0)
        
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        self.data.qpos[0] = np.random.uniform(-0.1, 0.1)
        self.data.qpos[1] = np.pi + np.random.uniform(-0.1, 0.1)
        
        self.data.qvel[:] = np.random.uniform(-0.1, 0.1, size=self.model.nv)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}