import numpy as np
import pygame
import torch
from torch import nn
from discreteEnv import EndlessCartPoleEnv

MODEL_PATH = "latest_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        return self.model(x)

def select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return policy_net(state).argmax(1).item()

def run_dqn(model_path=MODEL_PATH):
    pygame.init()
    env = EndlessCartPoleEnv(render_mode="human", initial_angle=0)
    state, _ = env.reset()
    clock = pygame.time.Clock()

    # Load trained DQN model
    policy_net = DQN().to(DEVICE)
    policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy_net.eval()
    print(f"DQN model loaded from {model_path}")

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
                    mode = "AUTO (DQN)" if auto_mode else "MANUAL"
                    print(f"Switched to {mode} mode")

        keys = pygame.key.get_pressed()
        action = None

        if not auto_mode:
            if keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
            else:
                action = None 
        else:
            action = select_action(policy_net, state)

        step_result = env.step(action)
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result

        episode_reward += reward
        env.render()
        clock.tick(60)

        if done:
            print(f"Episode {episode_count} | Total reward: {episode_reward:.2f}")
            state, _ = env.reset()
            episode_count += 1
            if auto_mode:
                total_reward += episode_reward
                total_episode += 1
            episode_reward = 0

    if auto_mode and total_episode > 0:
        print(f"Average reward over {total_episode} episodes in AUTO mode: {total_reward / total_episode:.2f}")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_dqn()
