import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from furuta_pendulum_rl import constants

class FurutaEnv(gym.Env):
    def __init__(self, model_path=constants.PENDULUM_MODEL_PATH, frame_skip=5, max_episode_steps=1000, pendulum_body_name="pendulum"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self.pendulum_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pendulum_body_name)

        self.action_space = spaces.Box(
            self.model.actuator_ctrlrange[:, 0].astype(np.float32), 
            self.model.actuator_ctrlrange[:, 1].astype(np.float32),
        )

        obs_low = np.array([0, 0, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([2 * np.pi, 2 * np.pi, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, (4,))

        self.poke_probability = 0.004
        self.poke_force_range = (0.05, 0.25)
        self.poke_duration = 2
        self.upright_position_threshold = 0.2

        self.poke_steps_remaining = 0
        self.current_poke = np.zeros(3)

    def _get_obs(self):
        normalized_qpos = self._normalize_angle(self.data.qpos)
        return np.concatenate([normalized_qpos, self.data.qvel]).astype(np.float32)

    #[0, 2pi)
    def _normalize_angle(self, theta):
        return theta % (2 * np.pi)

    def _is_upright(self):
        pendulum_angle = self._normalize_angle(self.data.qpos[1])
        return abs(pendulum_angle - np.pi) < self.upright_position_threshold

    def _apply_poke(self):
        if self.poke_steps_remaining > 0:
            self.poke_steps_remaining -= 1
        elif self._is_upright() and np.random.rand() < self.poke_probability:
            self.poke_steps_remaining = self.poke_duration

            base_angle = self.data.qpos[0]
            magnitude = np.random.uniform(*self.poke_force_range)
            direction = np.random.choice([-1.0, 1.0])

            self.current_poke = direction * magnitude * np.array([-np.sin(base_angle), np.cos(base_angle), 0.0])
        else:
            self.current_poke = np.zeros(3)

        self.data.xfrc_applied[self.pendulum_body_id, :3] = self.current_poke

    def step(self, action):
        self._apply_poke()

        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        base_angle = self.data.qpos[0]
        pendulum_angle = self.data.qpos[1]
        
        angle_diff = pendulum_angle - np.pi

        angle_reward = np.cos(angle_diff)
        base_penalty = 0.015 * (1.0 - np.cos(base_angle))
        action_penalty = 0.0005 * np.sum(np.square(action))
        velocity_penalty = 0.001 * np.sum(np.square(self.data.qvel))

        reward = angle_reward - base_penalty - action_penalty - velocity_penalty

        terminated = False
        
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        self.poke_steps_remaining = 0
        self.current_poke = np.zeros(3)
        self.data.xfrc_applied[:] = 0.0

        random = np.random.rand()
        if random < 0.5:
            self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
            self.data.qpos[1] = np.random.uniform(-0.2, 0.2)
            self.data.qvel[:] = np.random.uniform(-0.2, 0.2, size=self.model.nv)
        elif random < 0.75:
            self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
            self.data.qpos[1] = np.pi + np.random.uniform(-0.35, 0.35)
            self.data.qvel[:] = np.random.uniform(-0.5, 0.5, size=self.model.nv)
        else:
            self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
            self.data.qpos[1] = np.random.uniform(-np.pi, np.pi)
            self.data.qvel[0] = np.random.uniform(-3.0, 3.0)
            self.data.qvel[1] = np.random.uniform(-12.0, 12.0)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}