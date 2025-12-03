# environment.py
import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class TrainEnv(CartPoleEnv):
    def __init__(
        self,
        angle_sampler=None,
        render_mode=None,
        max_force=10.0,
        max_episode_steps=500,
        termination_angle=np.pi,            # radians; full rotation allowed
        success_angle=0.0873,               # 5° for evaluation
        success_steps_required=50,          # for evaluation
        success_angle_train=0.26,           # 15° for training
        success_steps_required_train=20
    ):
        super().__init__(render_mode=render_mode)

        # Initial pole angle sampler
        self.angle_sampler = angle_sampler if angle_sampler else (lambda: np.random.uniform(-np.pi, np.pi))

        # Continuous action space
        self.max_force = float(max_force)
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        # Observation space: [x, x_dot, cos(theta), sin(theta), theta_dot]
        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Episode control
        self.max_episode_steps = int(max_episode_steps)
        self.current_step = 0

        # Termination / success parameters
        self.termination_angle = float(termination_angle)
        self.success_angle = float(success_angle)
        self.success_steps_required = int(success_steps_required)

        # Lenient training success parameters
        self.success_angle_train = float(success_angle_train)
        self.success_steps_required_train = int(success_steps_required_train)

        # Internal counter for stable steps
        self._stable_steps = 0

    def reset(self, seed=None, options=None):
        # Small random perturbations for x, x_dot, theta_dot
        x = np.random.uniform(-0.05, 0.05)
        x_dot = np.random.uniform(-0.05, 0.05)
        theta = float(self.angle_sampler())
        theta_dot = np.random.uniform(-0.05, 0.05)

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step = 0
        self._stable_steps = 0

        return self._get_obs(), {}

    def step(self, action, training=True):
        x, x_dot, theta, theta_dot = self.state

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_force, self.max_force)
        force = float(action[0])

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / (
            self.masscart + self.masspole
        )
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

        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.termination_angle)
        truncated = bool(self.current_step >= self.max_episode_steps)

        x_norm = min(abs(x) / self.x_threshold, 1.0)
        angle_norm = min(abs(theta) / np.pi, 1.0)
        angle_term = 1.0 - angle_norm       # encourages moving toward upright
        dist_term = 1.0 - x_norm
        vel_penalty = 0.01 * (x_dot**2 + theta_dot**2)
        force_penalty = 0.001 * (force**2)
        base_reward = 0.8 * angle_term + 0.2 * dist_term
        reward = base_reward - vel_penalty - force_penalty

        if abs(theta) < np.deg2rad(5.0) and abs(x) < 0.1:
            reward += 1.0
        if terminated:
            reward -= 3.0

        if training:
            success_angle_use = self.success_angle_train
            success_steps_use = self.success_steps_required_train
        else:
            success_angle_use = self.success_angle
            success_steps_use = self.success_steps_required

        if abs(theta) < success_angle_use:
            self._stable_steps += 1
        else:
            self._stable_steps = 0

        is_success = self._stable_steps >= success_steps_use
        info = {"is_success": bool(is_success)}

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def set_stage_params(self, termination_angle=None, success_angle=None,
                         success_steps_required=None, angle_sampler=None):
        if termination_angle is not None:
            self.termination_angle = float(termination_angle)
        if success_angle is not None:
            self.success_angle = float(success_angle)
        if success_steps_required is not None:
            self.success_steps_required = int(success_steps_required)
        if angle_sampler is not None:
            self.angle_sampler = angle_sampler
