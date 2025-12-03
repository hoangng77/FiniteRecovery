import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class TrainEnv(CartPoleEnv):
    """
    Extended CartPole environment with:
      - Continuous action (force/torque) in a Box
      - Configurable termination angle and success criteria
      - Extra reward when pole is nearly upright and cart near center
      - Explicit 'is_success' in info dict
      - Observation vector: [x, x_dot, cos(theta), sin(theta), theta_dot]
    """

    def __init__(
        self,
        angle_sampler=None,
        render_mode=None,
        max_force=10.0,
        max_episode_steps=500,
        termination_angle=np.pi,
        success_angle=np.deg2rad(5.0),
        success_steps_required=50
    ):
        super().__init__(render_mode=render_mode)

        # Function to sample initial pole angle
        self.angle_sampler = angle_sampler if angle_sampler else (lambda: np.random.uniform(-np.pi, np.pi))

        # Continuous torque limits
        self.max_force = float(max_force)
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        # Observation: include cos(theta) and sin(theta) for continuity
        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Episode control
        self.max_episode_steps = int(max_episode_steps)
        self.current_step = 0

        # Curriculum / success parameters
        self.termination_angle = float(termination_angle)
        self.success_angle = float(success_angle)
        self.success_steps_required = int(success_steps_required)

        # Internal counter for stability detection
        self._stable_steps = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment state. Randomizes x, x_dot, theta, theta_dot.
        Returns (observation, info) as expected by Gymnasium.
        """
        x = np.random.uniform(-0.05, 0.05)
        x_dot = np.random.uniform(-0.05, 0.05)
        theta = float(self.angle_sampler())
        theta_dot = np.random.uniform(-0.05, 0.05)

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step = 0
        self._stable_steps = 0

        return self._get_obs(), {}

    def step(self, action):
        """
        Apply a physics step with continuous action.
        Returns: observation, reward, terminated, truncated, info
        """
        x, x_dot, theta, theta_dot = self.state

        # Ensure scalar action and clip to allowed force
        force = np.clip(action, -self.max_force, self.max_force)
        force = float(force[0])

        # --- Physics dynamics (from Gym CartPole) ---
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.masspole * self.length * theta_dot ** 2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / (self.masscart + self.masspole))
        )
        xacc = temp - self.masspole * self.length * thetaacc * costheta / (self.masscart + self.masspole)

        # Integrate using time step tau
        tau = self.tau
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # --- Base reward shaping ---
        reward = float(np.cos(theta) - 0.01 * (x ** 2 + x_dot ** 2 + theta_dot ** 2))

        # Episode bookkeeping
        self.current_step += 1
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.termination_angle)
        truncated = bool(self.current_step >= self.max_episode_steps)

        # Track stability for success detection
        if abs(theta) < self.success_angle:
            self._stable_steps += 1
        else:
            self._stable_steps = 0
        is_success = self._stable_steps >= self.success_steps_required
        info = {"is_success": bool(is_success)}

        # --- Extra reward modifications ---
        # Extra reward if pole is nearly upright and cart is near center
        if abs(theta) < np.deg2rad(5.0) and abs(x) < 0.1:
            reward += 1.0
        # Penalty if terminated
        if terminated:
            reward -= 3.0

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Return observation vector for agent."""
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def set_stage_params(self, termination_angle=None, success_angle=None, success_steps_required=None, angle_sampler=None):
        """Allow changing environment parameters without recreating the env."""
        if termination_angle is not None:
            self.termination_angle = float(termination_angle)
        if success_angle is not None:
            self.success_angle = float(success_angle)
        if success_steps_required is not None:
            self.success_steps_required = int(success_steps_required)
        if angle_sampler is not None:
            self.angle_sampler = angle_sampler
