import numpy as np
import pygame
from stable_baselines3 import SAC
from environment import NormalEnv

def run_sac(model_path="model"):
    pygame.init()

    env = NormalEnv(render_mode="human", angle_sampler=lambda: np.random.uniform(-np.pi/3, np.pi/3))

    state, _ = env.reset()
    clock = pygame.time.Clock()

    print(f"Loading SAC model from: {model_path}")
    model = SAC.load(model_path)
    print("SAC model loaded successfully.")

    auto_mode = False
    running = True
    episode_reward = 0
    episode_count = 0
    total_reward = 0
    total_episode = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                    mode = "AUTO (SAC)" if auto_mode else "MANUAL"
                    print(f"Switched to {mode} mode")

        keys = pygame.key.get_pressed()
        action = None

        if not auto_mode:
            if keys[pygame.K_LEFT]:
                action = np.array([-10.0])
            elif keys[pygame.K_RIGHT]:
                action = np.array([10.0])
            else:
                action = np.array([0.0])
        else:
            action, _ = model.predict(state, deterministic=True)

        step_result = env.step(action)
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
        else:
            state, reward, done, _ = step_result
            terminated, truncated = done, False

        episode_reward += reward

        env.render()
        clock.tick(60)

        if terminated:
            print(f"Episode {episode_count} | Total reward: {episode_reward:.2f}")
            state, _ = env.reset()
            episode_count += 1
            print(f"Starting Episode {episode_count} | Initial state: {np.round(state, 4)}")
            episode_reward = 0
        
        if auto_mode and terminated:
            total_reward += reward
            total_episode += 1

        if not running:
            print(f"Episode {episode_count} | Total reward: {episode_reward:.2f}")
            print(f"Average reward over {total_episode} episodes in AUTO mode: {total_reward / total_episode:.2f}")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_sac()
