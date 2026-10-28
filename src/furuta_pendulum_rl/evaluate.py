import numpy as np
import mujoco
from mujoco import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from furuta_pendulum_rl import furuta_env
from furuta_pendulum_rl import constants

def main():
    model = PPO.load(constants.RL_MODEL_PATH)
    
    env = DummyVecEnv([lambda: furuta_env.FurutaEnv()])
    env = VecNormalize.load(constants.VEC_NORMALIZE_PATH, env)
    env.training = False
    env.norm_reward = False

    mujoco_env = env.envs[0]
    model_mj = mujoco_env.model
    data_mj = mujoco_env.data

    env.reset()

    def controller(m, d):
        obs = mujoco_env._get_obs()
        obs_normalized = env.normalize_obs(obs)
        action, _ = model.predict(obs_normalized, deterministic=True)
        d.ctrl[:] = action

    mujoco.set_mjcb_control(controller)

    print("Start MuJoCo Viewer")
    viewer.launch(model_mj, data_mj)

if __name__ == "__main__":
    main()