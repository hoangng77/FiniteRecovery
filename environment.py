import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class FaultyCartPoleEnv(CartPoleEnv):
    def __init__(self, alpha_left=1.0, alpha_right=1.0,
                 friction=0.0, x_ext=0.0, delta_theta=0.0, tau=0.02, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.alpha_left = alpha_left
        self.alpha_right = alpha_right
        self.friction = friction
        self.x_threshold += x_ext
        self.tau = tau
        self.delta_theta = delta_theta
        self.original_screen_width = 600  # From original CartPoleEnv

    def render(self):
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            scale = self.original_screen_width / (2 * 2.4)
            new_width = int(scale * (2 * self.x_threshold))
            self.screen_width = new_width
            self.screen = pygame.display.set_mode((self.screen_width, 400))
        
        world_width = self.x_threshold * 2
        scale = self.original_screen_width / (2 * 2.4)
        self.scale = scale
        self.screen_width = int(world_width * scale)
        return super().render()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        if action is not None:
            force = self.force_mag if action == 1 else -self.force_mag
            force *= self.alpha_right if action == 1 else self.alpha_left
        else:
            force = 0.0

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole))
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / (self.masscart + self.masspole)

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * (xacc - self.friction * x_dot)
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        theta += self.delta_theta

        self.state = (x, x_dot, theta, theta_dot)

        reward = 1.0
        terminated = False
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

class EndlessCartPoleEnv(CartPoleEnv):
    def step(self, action=None):
        x, x_dot, theta, theta_dot = self.state

        if action is not None:
            force = self.force_mag if action == 1 else -self.force_mag
        else:
            force = 0.0

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole))
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / (self.masscart + self.masspole)

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        reward = 1.0
        terminated = False
        truncated = False
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}
