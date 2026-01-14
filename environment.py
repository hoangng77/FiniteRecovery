import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class FaultyEnv(CartPoleEnv):
    def __init__(self, alpha_left=1.0, alpha_right=1.0,
                 friction=0.0, x_ext=0.0, delta_theta=0.0, tau=0.02, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.alpha_left = alpha_left
        self.alpha_right = alpha_right
        self.friction = friction
        self.x_threshold += x_ext
        self.tau = tau
        self.delta_theta = delta_theta
        self.original_screen_width = 600

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
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole))
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

class NormalEnv(CartPoleEnv):
    def __init__(
        self,
        angle_sampler=None,
        render_mode=None,
        max_force=10.0,
        max_episode_steps=500,
        termination_angle=np.pi,
        success_angle=0.0873,
        success_steps_required=50
    ):
        super().__init__(render_mode=render_mode)
        self.angle_sampler = angle_sampler if angle_sampler else (lambda: np.random.uniform(-np.pi, np.pi))
        self.max_force = float(max_force)
        self.max_episode_steps = int(max_episode_steps)
        self.current_step = 0

        # Continuous action space
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        # Observation: [x, x_dot, cos(theta), sin(theta), theta_dot]
        high = np.array([np.finfo(np.float32).max]*5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Termination / success
        self.termination_angle = float(termination_angle)
        self.success_angle = float(success_angle)
        self.success_steps_required = int(success_steps_required)

        # Counter for consecutive upright steps
        self._stable_steps = 0
        self.state = None

    def reset(self, seed=None, options=None):
        x = np.random.uniform(-0.05, 0.05)
        x_dot = np.random.uniform(-0.05, 0.05)
        theta = float(self.angle_sampler())
        theta_dot = np.random.uniform(-0.05, 0.05)

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step = 0
        self._stable_steps = 0

        return self._get_obs(), {}

    def step(self, action, training=False):
        x, x_dot, theta, theta_dot = self.state

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_force, self.max_force)
        force = float(action[0])

        # Physics (same as CartPole)
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

        self.current_step += 1
        terminated = bool(abs(x) > self.x_threshold)
        truncated = bool(self.current_step >= self.max_episode_steps)

        # Reward shaping (matches TrainEnv)
        x_norm = min(abs(x)/self.x_threshold, 1.0)
        angle_norm = min(abs(theta)/np.pi, 1.0)
        angle_term = 1.0 - angle_norm
        dist_term = 1.0 - x_norm
        vel_penalty = 0.01 * (x_dot**2 + theta_dot**2)
        force_penalty = 0.001 * (force**2)
        reward = 0.8 * angle_term + 0.2 * dist_term - vel_penalty - force_penalty

        # Small upright bonus
        if abs(theta) < np.deg2rad(5.0) and abs(x) < 0.1:
            reward += 1.0
        if terminated:
            reward -= 3.0

        # Success tracking (like TrainEnv)
        if abs(theta) < self.success_angle:
            self._stable_steps += 1
        else:
            self._stable_steps = 0
        is_success = self._stable_steps >= self.success_steps_required

        info = {"is_success": bool(is_success)}
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def set_angle_sampler(self, angle_sampler):
        """Optional: update the starting angle sampler (for curriculum/testing)"""
        self.angle_sampler = angle_sampler