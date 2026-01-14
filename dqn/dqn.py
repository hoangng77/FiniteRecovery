import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment import EndlessCartPoleEnv

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=5000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episodes=500, batch_size=32, print_every=1):
    env = EndlessCartPoleEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        for step in range(200):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.replay(batch_size=batch_size)
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        if ep % print_every == 0:
            avg_reward = np.mean(reward_history[-20:])  # running average
            print(f"Episode {ep}, Total Reward: {total_reward}, Avg Reward(20): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}", flush=True)

    torch.save(agent.model.state_dict(), "dqn_cartpole.pth")
    print("Training finished and model saved.", flush=True)

if __name__ == "__main__":
    train_dqn()
