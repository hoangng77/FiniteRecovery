# train.py
import os
import time
import numpy as np
import multiprocessing

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

from environment import TrainEnv

# ---------------------------
# Curriculum / training config
# ---------------------------
angle_degrees = [30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 120, 150, 180]
angle_stages = [np.deg2rad(d) for d in angle_degrees]

success_threshold = 0.85
eval_episodes = 20
train_timesteps_per_chunk = 10000
n_envs = 16
total_timesteps_limit = int(1e8)

sac_kwargs = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    gamma=0.98,
    buffer_size=300_000,
    batch_size=256,
    tau=0.01,
    verbose=1,
)

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints",
    name_prefix="sac_curriculum"
)

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./newmodels", exist_ok=True)

# ---------------------------
# Environment helpers
# ---------------------------
def make_sampler(max_angle):
    """Uniform sampler in [-max_angle, max_angle]."""
    return lambda: float(np.random.uniform(-max_angle, max_angle))

def make_env(max_angle):
    """Create a single TrainEnv instance wrapped in Monitor."""
    sampler = make_sampler(max_angle)
    env = TrainEnv(angle_sampler=sampler)
    env = Monitor(env)
    return env

def make_env_factory(max_angle):
    return lambda: make_env(max_angle)

def make_subproc_vec_env(max_angle, n_envs=n_envs):
    return SubprocVecEnv([make_env_factory(max_angle) for _ in range(n_envs)])

# ---------------------------
# Replay buffer reset
# ---------------------------
def reset_replay_buffer(model):
    n_envs_current = model.get_env().num_envs
    model.replay_buffer = ReplayBuffer(
        buffer_size=int(model.buffer_size),
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        n_envs=n_envs_current
    )

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_policy_with_vecnormalize(model, n_eval=eval_episodes):
    vec_env = model.get_env()
    vec_env.training = False
    vec_env.norm_reward = False

    n_envs_local = vec_env.num_envs
    successes = 0
    episodes_done = 0
    episode_steps = np.zeros(n_envs_local, dtype=int)

    obs = vec_env.reset()
    max_steps = getattr(vec_env, "max_episode_steps", 500)

    while episodes_done < n_eval:
        actions, _ = model.predict(obs, deterministic=True)
        step_result = vec_env.step(actions)

        if len(step_result) == 5:
            obs, rewards, terminations, truncations, infos = step_result
        else:
            obs, rewards, dones, infos = step_result
            terminations = dones
            truncations = np.zeros_like(dones, dtype=bool)

        episode_steps += 1
        done_mask = np.logical_or(terminations, truncations)
        for i, done in enumerate(done_mask):
            if done:
                info = infos[i] if infos is not None else {}
                is_success = bool(info.get("is_success", False))
                length_success = (episode_steps[i] >= max_steps)
                if is_success or length_success:
                    successes += 1
                episodes_done += 1
                episode_steps[i] = 0
                if episodes_done >= n_eval:
                    break

    vec_env.training = True
    vec_env.norm_reward = False
    return successes / float(max(1, n_eval))

# ---------------------------
# Main training loop
# ---------------------------
def main():
    print("Creating initial SubprocVecEnv and VecNormalize (stage 0)...")
    raw_vec = make_subproc_vec_env(angle_stages[0], n_envs=n_envs)
    vec_norm = VecNormalize(raw_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    print("Creating SAC model...")
    model = SAC(
        policy=sac_kwargs["policy"],
        env=vec_norm,
        learning_rate=sac_kwargs["learning_rate"],
        gamma=sac_kwargs["gamma"],
        buffer_size=sac_kwargs["buffer_size"],
        batch_size=sac_kwargs["batch_size"],
        tau=sac_kwargs["tau"],
        verbose=sac_kwargs["verbose"],
    )

    total_trained = 0
    for stage_idx, max_angle in enumerate(angle_stages):
        print(f"\n=== Stage {stage_idx} / {len(angle_stages)-1} | max start angle ±{np.degrees(max_angle):.0f}° ===")
        new_raw_vec = make_subproc_vec_env(max_angle, n_envs=n_envs)

        try:
            vec_norm.venv.close()
        except Exception:
            pass
        vec_norm.venv = new_raw_vec
        model.set_env(vec_norm)
        reset_replay_buffer(model)

        stage_start_time = time.time()
        while True:
            model.learn(
                total_timesteps=train_timesteps_per_chunk,
                reset_num_timesteps=False,
                callback=checkpoint_cb
            )
            total_trained += train_timesteps_per_chunk

            success_rate = evaluate_policy_with_vecnormalize(model, n_eval=eval_episodes)
            elapsed = time.time() - stage_start_time
            print(f"[Stage {stage_idx}] trained timesteps: {total_trained} | elapsed: {elapsed/60:.2f} min | success rate: {success_rate:.3f}")

            if success_rate >= success_threshold:
                print(f"Stage {stage_idx} mastered (success {success_rate:.3f}). Saving model...")
                model.save(f"models/sac_stage_{stage_idx}_{int(np.degrees(max_angle))}deg")
                break

            if total_trained >= total_timesteps_limit:
                print("Reached total timesteps limit. Saving partial model and exiting.")
                model.save("models/sac_partial")
                return

    model.save("models/sac_full_range")
    print("\nTraining complete. Final model saved to models/sac_full_range.zip")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
