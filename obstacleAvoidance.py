import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random
import pickle
from collections import deque

# ====== CONSTANTS ======
LEARNING_RATE = 0.1       # Reduced from 0.3
DISCOUNT = 0.99
EPISODES = 1000           # Increased from 1000
SHOW_EVERY = 1
VELOCITY = 7
EPSILON = 0.9
EPSILON_DECAY = 0.999     # Slower decay (was 0.997)
MIN_EPSILON = 0.1         # Increased from 0.05
MAX_STEPS_PER_EPISODE = 500  # Prevent infinite episodes

# environment
class HuskyEnv:
    def __init__(self):
        self.client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Environment setup
        self.plane = p.loadURDF("plane.urdf")
        self.walls = self.setup_walls()
        self.car = p.loadURDF("husky/husky.urdf", basePosition=[0, 1, 0.1])
        
        self.goal_pos = np.array([2.5, 2.5])
        self.start_pos = np.array([0, 1])
        
        # ====== ENABLE ALL ACTIONS ======
        # 0=forward, 1=backward, 2=left, 3=right, 4=stop
        self.action_space = [0, 1, 2, 3, 4]
        
        # ====== IMPROVED STATE DISCRETIZATION ======
        # Smaller bins for better precision
        self.depth_bins = [0, 0.3, 0.5, 0.75, 1.0, 1.5, float('inf')]
        
        try:
            with open('qtable.pickle', 'rb') as f:
                self.q_table = pickle.load(f)
            print("Loaded existing Q-table")
        except:
            self.q_table = {}
            print("Created new Q-table")
    
    # adding walls to simulator
    def setup_walls(self):
        walls = []
        walls.append(create_wall([0, 0, 0.25], [0, 0, 0]))
        walls.append(create_wall([1, 0, 0.25], [0, 0, 0]))
        walls.append(create_wall([0, 2, 0.25], [0, 0, 0]))
        walls.append(create_wall([2, 0, 0.25], [0, 0, 0]))
        walls.append(create_wall([3, 1, 0.25], [0, 0, 1.57])) 
        walls.append(create_wall([3, 3, 0.25], [0, 0, 1.57])) 
        walls.append(create_wall([2, 4, 0.25], [0, 0, 0]))
        walls.append(create_wall([0, 4, 0.25], [0, 0, 0]))
        walls.append(create_wall([-1, 1, 0.25], [0, 0, 1.57])) 
        walls.append(create_wall([-1, 3, 0.25], [0, 0, 1.57])) 
        return walls
    
    def reset(self):
        orientation_quat = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.car, [0, 1, 0.1], orientation_quat)
        p.resetBaseVelocity(self.car, [0, 0, 0], [0, 0, 0])
        return self.get_state()
    
    # ====== STATE REPRESENTATION ======
    def get_state(self):
        # Get depth image with multiple viewpoints
        _, depth = get_camera_image(self.car)
        
        # Sample multiple points in the depth image (left, center, right)
        center_y = depth.shape[0] // 2
        left_x = depth.shape[1] // 4
        center_x = depth.shape[1] // 2
        right_x = 3 * depth.shape[1] // 4
        
        left_distance = depth[center_y][left_x]
        center_distance = depth[center_y][center_x]
        right_distance = depth[center_y][right_x]
        
        # Discretize the distances
        left_state = 0
        center_state = 0
        right_state = 0
        
        for i, threshold in enumerate(self.depth_bins[:-1]):
            if left_distance < self.depth_bins[i+1]:
                left_state = i
                break
        
        for i, threshold in enumerate(self.depth_bins[:-1]):
            if center_distance < self.depth_bins[i+1]:
                center_state = i
                break
        
        for i, threshold in enumerate(self.depth_bins[:-1]):
            if right_distance < self.depth_bins[i+1]:
                right_state = i
                break
        
        # Get position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.car)
        x, y, _ = position
        
        # Calculate distance and direction to goal
        dx = round(x - self.goal_pos[0], 1)
        dy = round(y - self.goal_pos[1], 1)
        
        # Get orientation as yaw angle (rotation around z-axis)
        _, _, yaw = p.getEulerFromQuaternion(orientation)
        # Discretize orientation to 8 directions (N, NE, E, SE, S, SW, W, NW)
        yaw_state = round((yaw + np.pi) * 4 / np.pi) % 8
        
        return (left_state, center_state, right_state, dx, dy, yaw_state)
    
    # ====== STEP FUNCTION WITH BETTER REWARDS ======
    def step(self, action):
        # Get current position BEFORE taking action
        current_pos, _ = p.getBasePositionAndOrientation(self.car)
        current_distance = np.linalg.norm(np.array([current_pos[0], current_pos[1]]) - self.goal_pos)
        
        # Take action
        if action == 0:
            move_forward(self.car, VELOCITY)
        elif action == 1:  # Re-enabled backward movement
            move_backward(self.car, VELOCITY)
        elif action == 2:  
            move_left(self.car, VELOCITY)
        elif action == 3:  
            move_right(self.car, VELOCITY)
        else: 
            move_husky(self.car, 0)
        
        # Run simulation
        for _ in range(10):  
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Get new position AFTER action
        new_pos, _ = p.getBasePositionAndOrientation(self.car)
        new_state = self.get_state()
        
        # Calculate new distance to goal
        x, y, _ = new_pos
        new_distance = np.linalg.norm(np.array([x, y]) - self.goal_pos)
        
        # ====== REWARD STRUCTURE ======
        # Base small penalty for each step to encourage efficiency
        reward = -0.1
        
        # Check if goal reached
        if new_distance < 0.2:
            reward = 100
            done = True
            print("GOAL REACHED!")
        else:
            done = False
            
            # Calculate and reward progress toward goal
            distance_improvement = current_distance - new_distance
            
            # More significant reward for moving toward goal
            reward += distance_improvement * 10
            
            # Small penalty for moving away from goal
            if distance_improvement < 0:
                reward -= 1
                
            # Check for collision - use improved collision detection
            if detect_collision(self.car, self.walls):
                reward = -20
                done = True
                print("COLLISION DETECTED")
        
        return new_state, reward, done
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        return self.q_table[state][action]
    
    def save_q_table(self):
        with open('qtable.pickle', 'wb') as f:
            pickle.dump(self.q_table, f)

# ====== COLLISION DETECTION ======
def detect_collision(car, walls):
    # Check using depth image
    image, depth = get_camera_image(car)
    center_y = depth.shape[0] // 2
    center_x = depth.shape[1] // 2
    
    # Check front area for obstacles
    min_distance = float('inf')
    for y in range(center_y-10, center_y+10):
        for x in range(center_x-10, center_x+10):
            if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                min_distance = min(min_distance, depth[y][x])
    
    # Also check for direct contact with walls using PyBullet
    contact_points = []
    for wall in walls:
        contact_points.extend(p.getContactPoints(car, wall))
    
    return min_distance < 0.5 or len(contact_points) > 0

def create_wall(position, orientation, size=(1, 0.1, 0.25), mass=0):
    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size)
    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    body = p.createMultiBody(baseMass=mass,
                           baseCollisionShapeIndex=collision_shape_id,
                           baseVisualShapeIndex=visual_shape_id,
                           basePosition=position,
                           baseOrientation=p.getQuaternionFromEuler(orientation))
    p.changeDynamics(body, -1, lateralFriction=0.9)
    return body

def get_camera_image(car):
    car_pos, car_orn = p.getBasePositionAndOrientation(car)
    rot_matrix = p.getMatrixFromQuaternion(car_orn)
    forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

    cam_pos = [car_pos[0] + 0.5 * forward_vec[0],
               car_pos[1] + 0.5 * forward_vec[1],
               car_pos[2] + 0.6]
    cam_target = [car_pos[0] + forward_vec[0],
                  car_pos[1] + forward_vec[1],
                  car_pos[2] + 0.3]

    view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100)

    _, _, rgba_img, depth_img, _ = p.getCameraImage(160, 120, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    img = np.reshape(rgba_img, (120, 160, 4))[:, :, :3]
    depth_array = np.array(depth_img).reshape((120, 160))
    return img, depth_array

def move_husky(car, velocity):
    for joint in [2, 3, 4, 5]:
        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity=velocity, force=1000)

def move_left(car, velocity):
    p.setJointMotorControl2(car, 2, p.VELOCITY_CONTROL, targetVelocity=-(velocity * 1.1), force=1000)
    p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, targetVelocity=-(velocity * 1.1), force=1000)
    p.setJointMotorControl2(car, 3, p.VELOCITY_CONTROL, targetVelocity=velocity * 2, force=1000)
    p.setJointMotorControl2(car, 5, p.VELOCITY_CONTROL, targetVelocity=velocity * 2, force=1000)

def move_forward(car, velocity):
    for joint in [2, 3, 4, 5]:  
        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity=velocity, force=1000)

def move_backward(car, velocity):
    for joint in [2, 3, 4, 5]: 
        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity=-velocity, force=1000)

def move_right(car, velocity):
    p.setJointMotorControl2(car, 2, p.VELOCITY_CONTROL, targetVelocity=velocity, force=1000)
    p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, targetVelocity=velocity, force=1000)
    p.setJointMotorControl2(car, 3, p.VELOCITY_CONTROL, targetVelocity=-(velocity * 1.1), force=1000)
    p.setJointMotorControl2(car, 5, p.VELOCITY_CONTROL, targetVelocity=-(velocity * 1.1), force=1000)

def is_Obstacle(car):
    image, depth = get_camera_image(car)
    center_y = depth.shape[0] // 2
    center_x = depth.shape[1] // 2
    distance = depth[center_y][center_x]
    return distance < 0.75

# ====== TRAINING FUNCTION WITH DEBUGGING ======
def train_agent():
    env = HuskyEnv()
    epsilon = EPSILON
    episode_rewards = []
    success_count = 0  # Track successful episodes
    steps_to_goal = []  # Track steps needed to reach goal
    
    # Visualize the goal
    goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 0.7])
    goal_body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_visual,
        basePosition=[env.goal_pos[0], env.goal_pos[1], 0.2]
    )
    
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        success = False
        
        # For debugging - show episode progress
        if episode % SHOW_EVERY == 0:
            print(f"\nStarting Episode {episode}...")
            debug_mode = True
        else:
            debug_mode = False
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = random.choice(env.action_space)
                if debug_mode and steps % 10 == 0:
                    print(f"  Step {steps}: Random action {action}")
            else:
                action = np.argmax([env.get_q_value(state, a) for a in env.action_space])
                if debug_mode and steps % 10 == 0:
                    print(f"  Step {steps}: Best action {action}")
            
            # Take action
            new_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Debug info
            if debug_mode and steps % 10 == 0:
                position, _ = p.getBasePositionAndOrientation(env.car)
                x, y, z = position
                distance_to_goal = np.linalg.norm(np.array([x, y]) - env.goal_pos)
                print(f"    Position: ({x:.2f}, {y:.2f}), Distance to goal: {distance_to_goal:.2f}, Reward: {reward:.2f}")
            
            # Update Q-table
            current_q = env.get_q_value(state, action)
            max_future_q = np.max([env.get_q_value(new_state, a) for a in env.action_space])
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            # Set the new Q-value
            if state not in env.q_table:
                env.q_table[state] = np.zeros(len(env.action_space))
            env.q_table[state][action] = new_q
            
            state = new_state
            
            # Track successful episodes
            if done and reward > 0:
                success = True
                success_count += 1
                steps_to_goal.append(steps)
                if debug_mode:
                    print(f"  SUCCESS! Reached goal in {steps} steps with reward {episode_reward}")
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
        
        # Show progress
        if episode % SHOW_EVERY == 0:
            avg_reward = np.mean(episode_rewards[-SHOW_EVERY:])
            success_rate = (success_count / min(SHOW_EVERY, episode+1)) * 100 if episode > 0 else 0
            avg_steps = np.mean(steps_to_goal) if steps_to_goal else "N/A"
            
            print(f"Episode: {episode}/{EPISODES}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Avg Steps to Goal: {avg_steps}")
            print(f"  Epsilon: {epsilon:.3f}")
            
            # Reset counters for next batch
            success_count = 0
            steps_to_goal = []
    
    # Save final Q-table
    env.save_q_table()
    print("\nTraining complete!")
    return env

# Testing
def test_agent(env, num_tests=5):
    success_count = 0
    
    for test in range(num_tests):
        print(f"\nTest run {test+1}/{num_tests}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            
            # Choose best action from Q-table
            action = np.argmax([env.get_q_value(state, a) for a in env.action_space])
            print(f"Step {steps}: Taking action {action}")
            
            # Take action
            state, reward, done = env.step(action)
            total_reward += reward
            
            # Debug info
            position, _ = p.getBasePositionAndOrientation(env.car)
            x, y, z = position
            distance_to_goal = np.linalg.norm(np.array([x, y]) - env.goal_pos)
            print(f"  Position: ({x:.2f}, {y:.2f}), Distance to goal: {distance_to_goal:.2f}")
            
            # Success tracking
            if done and reward > 0:
                success_count += 1
                print(f"SUCCESS! Reached goal in {steps} steps")
            elif done:
                print(f"FAILED! Hit obstacle after {steps} steps")
            
            time.sleep(0.1)  # Slow down for visualization
        
        if not done:
            print(f"TIMEOUT after {steps} steps")
        
        print(f"Test {test+1} complete, reward: {total_reward:.2f}")
    
    success_rate = (success_count / num_tests) * 100
    print(f"\nOverall success rate: {success_rate:.1f}% ({success_count}/{num_tests})")

if __name__ == "__main__":
    # Training mode
    print("Starting training mode...")
    env = train_agent()
    
    # Testing mode
    print("\n=== TRAINING COMPLETE ===")
    input("Press Enter to run test mode...")
    test_agent(env, num_tests=5)
    
    # Keep simulation running
    print("\nTests complete. Simulation will keep running until you close the window.")
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
