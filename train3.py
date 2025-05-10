import pybullet as p
import pybullet_data
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from huskySimulation2 import move_forward, move_backward, move_left, move_right, get_camera_image

# Hyperparameters and Constants
LEARNING_RATE    = 0.001
DISCOUNT_FACTOR  = 0.99
MEMORY_SIZE      = 10000
BATCH_SIZE       = 64
EPSILON_START    = 1.0
EPSILON_DECAY    = 0.995
MIN_EPSILON      = 0.01
TARGET_UPDATE    = 10
TIME_STEP        = 1.0 / 120.0
MAX_STEPS        = 500
NUM_EPISODES     = 10
WINDOW_SIZE      = 50    # for running average
SHOW_EVERY       = 10   # episodes per metrics batch

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.memory)

# Build maze and set goal at (2.5, 2.5)
def create_maze_and_goal():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    wall_coords = [
        ([0,0,0.25],[0,0,0]), ([1,0,0.25],[0,0,0]), ([0,2,0.25],[0,0,0]),
        ([2,0,0.25],[0,0,0]), ([3,1,0.25],[0,0,1.57]), ([3,3,0.25],[0,0,1.57]),
        ([2,4,0.25],[0,0,0]), ([0,4,0.25],[0,0,0]), ([-1,1,0.25],[0,0,1.57]),
        ([-1,3,0.25],[0,0,1.57])
    ]
    for pos, orn in wall_coords:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=(1,0.1,0.25)),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=(1,0.1,0.25)),
            basePosition=pos,
            baseOrientation=p.getQuaternionFromEuler(orn)
        )
    car = p.loadURDF("husky/husky.urdf", basePosition=[0,1,0.1])
    goal = [2.5, 2.5]
    # Visualize goal
    goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0,1,0,0.7])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_vis, basePosition=[*goal,0.2])
    return car, goal

# Epsilon-greedy action selection
def select_action(state, epsilon, policy_net):
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        return torch.argmax(policy_net(torch.FloatTensor(state))).item()

# Compute reward, done flag, and success flag
def compute_reward_done_success(car, goal, step):
    pos, _ = p.getBasePositionAndOrientation(car)
    dist = np.linalg.norm(np.array(pos[:2]) - np.array(goal))
    reward = -dist
    # collision penalty
    _, depth = get_camera_image(car)
    if np.min(depth) < 0.5:
        reward -= 5.0
    reached = dist < 0.2
    if reached:
        reward += 100.0
    done = reached or (step >= MAX_STEPS)
    return reward, done, reached

# Execute an action
def execute_action(car, action):
    if action == 0:
        move_forward(car, 5)
    elif action == 1:
        move_backward(car, 5)
    elif action == 2:
        move_left(car, 5)
    elif action == 3:
        move_right(car, 5)

# Training routine with metric tracking
def train():
    p.connect(p.DIRECT)
    p.setGravity(0,0,-9.81)
    p.setTimeStep(TIME_STEP)

    car, goal = create_maze_and_goal()
    policy_net = DQN(input_dim=4, output_dim=4)
    target_net = DQN(input_dim=4, output_dim=4)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START

    episode_rewards = []
    running_avg = []
    success_count = 0
    steps_to_goal = []
    # metrics batches
    metrics_eps, metrics_avg_r, metrics_succ_r, metrics_avg_s = [], [], [], []

    for ep in range(NUM_EPISODES):
        # reset episode
        pos, _ = p.getBasePositionAndOrientation(car)
        state = np.array([pos[0], pos[1], goal[0], goal[1]])
        total_reward = 0
        step = 0
        done = False

        while not done:
            action = select_action(state, epsilon, policy_net)
            execute_action(car, action)
            p.stepSimulation()
            time.sleep(TIME_STEP)

            step += 1
            pos_new, _ = p.getBasePositionAndOrientation(car)
            next_state = np.array([pos_new[0], pos_new[1], goal[0], goal[1]])
            reward, done, reached = compute_reward_done_success(car, goal, step)
            total_reward += reward

            memory.push((state, action, reward, next_state, done))
            state = next_state

            # train DQN
            if len(memory) >= BATCH_SIZE:
                s, a, r, s2, d = memory.sample(BATCH_SIZE)
                states      = torch.FloatTensor(s)
                actions     = torch.LongTensor(a)
                rewards     = torch.FloatTensor(r)
                next_states = torch.FloatTensor(s2)
                dones       = torch.FloatTensor(d.astype(float))

                q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q = target_net(next_states).max(1)[0].detach()
                target_q = rewards + DISCOUNT_FACTOR * next_q * (1 - dones)

                loss = nn.MSELoss()(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if reached:
                success_count += 1
                steps_to_goal.append(step)

        # end episode
        episode_rewards.append(total_reward)
        running_avg.append(np.mean(episode_rewards[-WINDOW_SIZE:]))

        # decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # update target network
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # batch metrics
        if (ep + 1) % SHOW_EVERY == 0:
            avg_r = np.mean(episode_rewards[-SHOW_EVERY:])
            succ_r = (success_count / SHOW_EVERY) * 100
            avg_s = np.mean(steps_to_goal) if steps_to_goal else 0
            metrics_eps.append(ep + 1)
            metrics_avg_r.append(avg_r)
            metrics_succ_r.append(succ_r)
            metrics_avg_s.append(avg_s)
            print(f"Episode {ep+1}: Avg Reward={avg_r:.2f}, Success Rate={succ_r:.1f}%, Avg Steps={avg_s:.1f}")
            success_count = 0
            steps_to_goal = []

    p.disconnect()

    # Plot metrics
    plt.figure()
    plt.plot(metrics_eps, metrics_avg_r, marker='o')
    plt.title(f'Average Reward per {SHOW_EVERY} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_reward1.png')
    plt.close()

    plt.figure()
    plt.plot(metrics_eps, metrics_succ_r, marker='o')
    plt.title(f'Success Rate per {SHOW_EVERY} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('success_rate1.png')
    plt.close()

    plt.figure()
    plt.plot(metrics_eps, metrics_avg_s, marker='o')
    plt.title(f'Average Steps to Goal per {SHOW_EVERY} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Avg Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_steps.png')
    plt.close()

    #plt.show()

if __name__ == "__main__":
    train()
