from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from furuta_pendulum_rl import furuta_env
from furuta_pendulum_rl import constants

def main():
    env = make_vec_env(lambda: furuta_env.FurutaEnv(), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    print("Start training")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
    )

    model.learn(total_timesteps=1_000_000)

    model.save(constants.RL_MODEL_PATH)
    env.save(constants.VEC_NORMALIZE_PATH)
    print("Training finished")

if __name__ == "__main__":
    main()