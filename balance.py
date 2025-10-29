import numpy as np
import pygame
from environment import FaultyCartPoleEnv, EndlessCartPoleEnv

# Fixed deterministic policy is a simple proportional-derivative controller, counteracting the changes in pole angle.
# It only works for small angles and cannot recovers if pole falls over.

def fixed_policy(state, Kp=3.0, Kd=1.0):
    x, x_dot, theta, theta_dot = state
    control_signal = Kp * theta + Kd * theta_dot
    return 1 if control_signal > 0 else 0

# Linear quadratic regulator (LQR) policy for balancing the pole around the upright position. Cannot swingup from downward position

def lqr_policy(state, K=np.array([-10, -1, 100, 10])):
    control_signal = np.dot(K, state)
    return 1 if control_signal > 0 else 0

# This is a hybrid swing-up and balancing policy. So it used energy to bring the pole up when it is far from upright, and once reach upright it will start balancing using LQR.

def swingup_policy(state, K=np.array([-10, -1, 100, 10]), energy_gain=1.0):
    x, x_dot, theta, theta_dot = state
    g = 9.8
    l = 0.5
    m = 0.1
    energy = 0.5 * m * (l * theta_dot)**2 - m * g * l * np.cos(theta)
    desired_energy = 0
    force = energy_gain * (energy - desired_energy) * np.sign(theta_dot * np.cos(theta))
    if abs(theta) < 0.2:
        force = np.dot(K, state)
    return 1 if force > 0 else 0

def run_test():
    env = FaultyCartPoleEnv(
        alpha_left=0.8,
        alpha_right=1.2,
        friction=0.7,
        x_ext=3.0,
        delta_theta=0.0,
        tau=0.02,
        render_mode="human"
    )

    state, _ = env.reset()
    clock = pygame.time.Clock()
    running = True
    action = None  
    fixed = False
    lqr = False
    swingup = False
    auto = False
    rightCounter = 0
    leftCounter = 0

    def count(action):
        nonlocal rightCounter, leftCounter
        if action == 1:
            rightCounter += 1
        elif action == 0:
            leftCounter += 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    print("Switching to fixed deterministic mode")
                    fixed, lqr, swingup = True, False, False
                    auto = True
                elif event.key == pygame.K_2:
                    print("Switching to LQR mode")
                    fixed, lqr, swingup = False, True, False
                    auto = True
                elif event.key == pygame.K_3:
                    print("Switching to swing-up mode")
                    fixed, lqr, swingup = False, False, True
                    auto = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
            auto = False
        elif keys[pygame.K_RIGHT]:
            action = 1
            auto = False
        elif auto:
            if swingup:
                action = swingup_policy(state)
                count(action)
            elif lqr:
                action = lqr_policy(state)
                count(action)
            elif fixed:
                action = fixed_policy(state)
                count(action)
        else:
            action = None

        state, reward, done, _, _ = env.step(action)
        env.render()
        clock.tick(60)
    print(f"Right action count: {rightCounter}, Left action count: {leftCounter}")
    env.close()
    pygame.quit()

def run_normal():
    env = EndlessCartPoleEnv(render_mode="human")
    state, _ = env.reset()
    clock = pygame.time.Clock()
    running = True
    action = None  
    fixed = False
    lqr = False
    swingup = False
    auto = False
    rightCounter = 0
    leftCounter = 0

    def count(action):
        nonlocal rightCounter, leftCounter
        if action == 1:
            rightCounter += 1
        elif action == 0:
            leftCounter += 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    print("Switching to fixed deterministic mode")
                    fixed, lqr, swingup = True, False, False
                    auto = True
                elif event.key == pygame.K_2:
                    print("Switching to LQR mode")
                    fixed, lqr, swingup = False, True, False
                    auto = True
                elif event.key == pygame.K_3:
                    print("Switching to swing-up mode")
                    fixed, lqr, swingup = False, False, True
                    auto = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
            auto = False
        elif keys[pygame.K_RIGHT]:
            action = 1
            auto = False
        elif auto:
            if swingup:
                action = swingup_policy(state)
                count(action)
            elif lqr:
                action = lqr_policy(state)
                count(action)
            elif fixed:
                action = fixed_policy(state)
                count(action)
        else:
            action = None

        state, reward, done, _, _ = env.step(action)
        env.render()
        clock.tick(60)
    print(f"Right action count: {rightCounter}, Left action count: {leftCounter}")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_test()
