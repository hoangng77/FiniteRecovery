GitHub repo: https://github.com/hoangng77/FiniteRecovery
Environment:
TrainEnv is a customized CartPole environment designed for reinforcement learning with extreme initial pole angles and curriculum learning.
It extends the standard CartPole with:
Continuous actions in [-10, 10] to handle precise torque control, necessary for swing-up.
Configurable initial pole angles via angle_sampler(), enabling curriculum stages.
Flexible termination and success criteria: lenient for training, strict for evaluation.
Observations [x, x_dot, cos(theta), sin(theta), theta_dot], which provide smooth, continuous representations of the pole angle for neural networks (avoiding wrap-around discontinuities).
→ Using SAC is appropriate due to its continuous action support and exploration capabilities, making it suitable for learning this challenging control task.

Code:
Environment initialization:
def __init__(
        self,
        angle_sampler=None,
        render_mode=None,
        max_force=10.0,
        max_episode_steps=500,
        termination_angle=np.pi,            
        success_angle=0.0873,               
        success_steps_required=50,         
        success_angle_train=0.26,           
        success_steps_required_train=20
    ):
        super().__init__(render_mode=render_mode)

Define the variables, force range being [-10, 10], each episode will have 500 steps maximum, termination angle is pi (pole fall straight down), we define success by:
variation of 5 degree when evaluation
variation of 15 degree when training
can survive 50 steps consecutively while evaluation
can survive 20 steps consecutively while training
These can be changed in the future if we have a different approach.

2. Reset environment:
def reset(self, seed=None, options=None):
        # Small random perturbations for x, x_dot, theta_dot
        x = np.random.uniform(-0.05, 0.05)
        x_dot = np.random.uniform(-0.05, 0.05)
        theta = float(self.angle_sampler())
        theta_dot = np.random.uniform(-0.05, 0.05)

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32) // now initialize the state
        self.current_step = 0 // reset step counter
        self._stable_steps = 0 // this is used to count “success” steps

        return self._get_obs(), {} // get_obs is just returning the state

It added more random perturbations when reset to the x, x_dot, and theta_dot, since theta is fixed from angle_sampler, we will not touch it.
3. Dynamics and state 
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

Using dynamics equation and update state using Euler integration. We will terminate when the cart goes out of bound and truncate if we take more steps than an episode step.
4. Reward function
x_norm = min(abs(x)/self.x_threshold, 1.0)
angle_norm = min(abs(theta)/np.pi, 1.0)
angle_term = 1.0 - angle_norm
dist_term = 1.0 - x_norm
vel_penalty = 0.01 * (x_dot**2 + theta_dot**2)
force_penalty = 0.001 * (force**2)
reward = 0.8 * angle_term + 0.2 * dist_term - vel_penalty - force_penalty

if abs(theta) < np.deg2rad(5.0) and abs(x) < 0.1:
reward += 1.0
if terminated:
reward -= 3.0

Reward shaping using normalized cart and pole positions, and all the penalites are to help balance the pole the most stable. Bonus reward for keeping it below 5 degree near upright and if fails, apply a penalty. 
→ After multiple training, it would be great to use energy-based reward, such as accounting for kinetic and potential energy so that the agent learns better through the reward function.


II. Training strategy
Algorithm choice: Soft Actor Critic for continuous action spaces, and since this problem is complex with various exploration, using SAC fits because of its attribute of encouraging exploration through. SAC is also a stochastic and sample-efficient algorithm. 
Architecture: SAC has 3 neural networks, the first one is Actor network to output probability distributions over actions and two Critic q-networks to estimate how good an action is on a given state (using 2 to prevent overestimation bias). It’s also off-policy, therefore keeping track of the replay buffer is really important. 
Training strategy: using adaptive learning starting at 30 degree up to 90 degree. Training this only relies heavily on the computation of the CartPole physics, not GPU-heavy (only 3 [512, 512, 256] networks), therefore we use parallel processing with SubProcVecEnv from SB3, each envs run an independent episode, experience from all envs are updated in a shared replay buffer.
Observation normalization: this is really crucial, without normalization, large-magnitude observations can dominate network activations, leading to unstable gradients and slow convergence. We therefore apply VecNormalize to normalize observations online using running statistics (mean and variance). Reward normalization is intentionally disabled to preserve the carefully designed reward shaping
Evaluation strategy: evaluation is performed periodically using the same normalized environment but with deterministic actions (no exploration noise) and frozen normalization statistics. Success is defined as maintaining the pole within a strict angular threshold for a fixed number of consecutive steps. This separation between training and evaluation criteria ensures that exploration does not artificially inflate performance metrics

Results from training:
Training takes really fast to complete 10 million timesteps, so timesteps are not a problem, we used to have positive results when trained on uniform distribution between (-training_angle, training_angle) (for instance training_angle = pi/4 stands for training at initial angle between -pi/4 and pi/4) and evaluation on that same distribution. However, when we switched to evaluating on exact training_angle, we failed miserably. 
I tested on the ProtoTwin cartpole environment instead, because the physics are more realistic compared to CartPole, using PPO, and succeed in swinging up pole starting from pi.

III. ProtoTwin CartPole environment: 
To further investigate the failure cases observed in the custom Gym CartPole environment, we experimented with the ProtoTwin CartPole simulator, which provides more realistic physics and actuation compared to the analytical CartPole model.

Environment characteristic: 
Unlike Gym CartPole, ProtoTwin uses a motor-driven cart with velocity control rather than direct force application. This introduces actuator dynamics, force limits, and more realistic interactions between the cart and the pole. As a result, the learned policy is constrained to operate within physically plausible limits.
The observations are obtained directly from ProtoTwin signal addresses and consist of: normalized cart position, cart velocity, normalized pole angle (with angle wrapping), pole angular velocity. 
The action space is continuous and consists of a single scalar action corresponding to the target cart velocity, clipped to [−1,1].
Reward function: The reward function is designed to prioritize pole stability while discouraging excessive movement and force usage: the agent is rewarded for keeping the pole close to the upright position, a penalty is applied when the cart moves away from the center, a small penalty discourages large motor forces
The final reward is scaled by the simulation timestep to ensure consistency across rollouts: r=Δt⋅(1−∣θ^∣−0.5⋅∣x^∣−0.0025⋅∣F∣)
Training setup: 
Algorithm: Proximal Policy Optimization (PPO)
Number of parallel environments: 100
Policy architecture:
Actor: [128, 64, 32]
Critic: [128, 64, 32]
Batch size: 10,000
Rollout length: 1,000 steps
Learning rate: quadratic decay schedule
Total training steps: 10 million









