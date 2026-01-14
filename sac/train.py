import os
import copy
import time
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environment import TrainEnv

N_ENVS = 16
TRAIN_CHUNK = 25000
TOTAL_TIMESTEPS = 1500000
EVAL_EPISODES = 10

SUCCESS_HIGH = 0.8
SUCCESS_LOW = 0.3
MAX_ANGLE = np.pi / 2  # 90 degrees

def make_env():
    return Monitor(TrainEnv())

def make_vec_env(n):
    return SubprocVecEnv([make_env for _ in range(n)])

def evaluate(model, eval_env, n_episodes=EVAL_EPISODES):
    obs = eval_env.reset()
    episodes = 0
    successes = 0

    while episodes < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = eval_env.step(action)

        for i, done in enumerate(dones):
            if done:
                episodes += 1
                # success if pole upright at end
                theta_cos = obs[i][2]
                successes += theta_cos > 0.95

    return successes / n_episodes

def main():
    os.makedirs("models", exist_ok=True)

    vec_env = VecNormalize(
        make_vec_env(N_ENVS),
        norm_obs=True,
        norm_reward=False,
    )

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        verbose=1,
    )

    # Evaluation env
    eval_env = VecNormalize(make_vec_env(1), norm_obs=True, norm_reward=False)
    eval_env.obs_rms = copy.deepcopy(vec_env.obs_rms)
    eval_env.training = False

    curriculum_angle = 0.0
    start_time = time.time()

    while True:
        # Set current curriculum difficulty
        vec_env.env_method("set_curriculum_angle", curriculum_angle)

        # Train chunk of timesteps
        model.learn(total_timesteps=TRAIN_CHUNK, reset_num_timesteps=False)

        # Evaluate policy
        success_rate = evaluate(model, eval_env, EVAL_EPISODES)
        elapsed = (time.time() - start_time) / 60
        print(f"Angle ±{np.degrees(curriculum_angle):.1f}°, success={success_rate:.2f}, elapsed={elapsed:.1f} min")

        # Adaptive update
        if success_rate > SUCCESS_HIGH:
            curriculum_angle = min(curriculum_angle * 1.1 + 0.01, MAX_ANGLE)
        elif success_rate < SUCCESS_LOW:
            curriculum_angle = max(curriculum_angle * 0.95, 0.0)

        # Stop if full swing-up learned
        if curriculum_angle >= MAX_ANGLE:
            break

    # Save final model
    model.save("models/cartpole")
    print("Training complete!")

if __name__ == "__main__":
    main()