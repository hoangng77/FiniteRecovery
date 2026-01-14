import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class EndlessCartPoleEnv(CartPoleEnv):
    def __init__(self, max_force=10.0, initial_angle=0.0, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.max_force = max_force
        self.initial_angle = initial_angle

        self.action_space = spaces.Discrete(2)

        high = np.array([np.finfo(np.float32).max]*4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.max_episode_steps = 500
        self.current_step = 0

    def reset(self, seed=None, options=None):
        x = 0.0
        x_dot = 0.0
        theta = self.initial_angle
        theta_dot = np.random.uniform(1.0, 2.0) * np.random.choice([-1, 1])

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action=None):
        x, x_dot, theta, theta_dot = self.state

        if action is not None:
            force = self.max_force if action == 1 else -self.max_force
        else:
            force = 0.0 

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole))
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / (self.masscart + self.masspole)

        tau = self.tau
        x += tau * x_dot
        x_dot += tau * xacc
        theta += tau * theta_dot
        theta_dot += tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        reward = 1.0

        terminated = abs(theta) > np.pi
        truncated = False

        self.current_step += 1
        return self.state, reward, terminated, truncated, {}
