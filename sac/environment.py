import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class TrainEnv(CartPoleEnv):
    """
    CartPole swing-up + balance environment.
    Observation: [x, x_dot, cos(theta), sin(theta), theta_dot]
    Action: continuous force
    """

    def __init__(self, render_mode=None, max_force=10.0, max_episode_steps=500):
        super().__init__(render_mode=render_mode)

        self.max_force = float(max_force)
        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(1,),
            dtype=np.float32,
        )

        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # curriculum
        self.curriculum_angle = 0.0

        # success tracking
        self.upright_steps = 0
        self.upright_threshold = 0.2  # rad

    def set_curriculum_angle(self, angle_rad):
        self.curriculum_angle = float(np.clip(angle_rad, 0.0, np.pi))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = np.random.uniform(-0.05, 0.05)
        x_dot = np.random.uniform(-0.05, 0.05)
        theta = np.random.uniform(-self.curriculum_angle, self.curriculum_angle)
        theta_dot = np.random.uniform(-0.05, 0.05)

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step = 0
        self.upright_steps = 0

        return self._get_obs(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(np.clip(action[0], -self.max_force, self.max_force))

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / (
            self.masscart + self.masspole
        )

        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole))
        )

        xacc = temp - self.masspole * self.length * thetaacc * costheta / (
            self.masscart + self.masspole
        )

        tau = self.tau
        x += tau * x_dot
        x_dot += tau * xacc
        theta += tau * theta_dot
        theta_dot += tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step += 1

        mgl = self.masspole * self.gravity * self.length
        E = mgl * (1 - np.cos(theta)) + 0.5 * (theta_dot * self.length) ** 2
        E_target = mgl

        r_energy = np.exp(-((E - E_target) / E_target) ** 2)
        r_upright = np.exp(-(theta / 0.25) ** 2)

        alpha = np.exp(-4.0 * abs(theta))
        reward = (1 - alpha) * r_energy + alpha * r_upright

        reward -= 0.01 * (x / self.x_threshold) ** 2
        reward -= 0.001 * force**2

        if abs(theta) < self.upright_threshold:
            self.upright_steps += 1
        else:
            self.upright_steps = 0

        terminated = bool(abs(x) > self.x_threshold)
        truncated = bool(self.current_step >= self.max_episode_steps)

        info = {
            "upright_steps": self.upright_steps,
            "theta": theta,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array(
            [x, x_dot, np.cos(theta), np.sin(theta), theta_dot],
            dtype=np.float32,
        )