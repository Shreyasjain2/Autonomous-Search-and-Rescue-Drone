import gym
import numpy as np
import pybullet as p
import pybullet_data
import os
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import math

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        
        # Define feature extractors for each observation component
        self.position_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.imu_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.coverage_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*18*18, 128)  # Adjust based on your grid size
        )
        
        self.combine_net = nn.Sequential(
            nn.Linear(64+64+128, 256),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        position_features = self.position_net(observations["position"])
        imu_features = self.imu_net(observations["imu"])
        
        # Process coverage map (add channel dimension)
        coverage_features = self.coverage_net(observations["coverage_map"].unsqueeze(1))
        
        combined = torch.cat([position_features, imu_features, coverage_features], dim=1)
        return self.combine_net(combined)

class SearchDroneEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render=True):
        super(SearchDroneEnv, self).__init__()
        self.render_mode = render
        self.search_area = [-50, 50, -50, 50]  # x_min, x_max, y_min, y_max
        self.grid_size = 5  # meters per grid cell
        self.fov_radius = 5  # detection radius
        self.fov_angle = 120  # degrees
        self.max_steps = 5000  # Increased from 2000 to 5000
        self.fixed_height = 2.0
        self.duck_scale = 3.0

        # PyBullet connection
        self.client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Action and Observation Spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'imu': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'coverage_map': spaces.Box(low=0, high=1, 
                shape=(self._get_grid_size(), self._get_grid_size()), dtype=np.float32)
        })
        
        # Initialize variables
        self.drone = None
        self.victims = []
        self.visited_cells = set()
        self.current_step = 0

    def _get_grid_size(self):
        return int((self.search_area[1]-self.search_area[0])/self.grid_size)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load environment
        plane_id = p.loadURDF("./data/plane.urdf", useMaximalCoordinates=True)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.5, 0.4, 0.3, 1])
        #p.loadSDF("./data/stadium.sdf")
        start_pos = [0, 0, 1]
        self.drone = p.loadURDF("./data/Quadrotor/quadrotor.urdf", start_pos, useFixedBase=False)
        # self.drone = p.loadURDF("./data/r2d2.urdf", start_pos, useFixedBase=False)
        
        self.fov_lines = []
        
        # Create initial FOV visualization
        self._update_fov_visualization()

        # Generate victims with visualization
        self.victims = []
        victim_sphere_radius = 0.5
        num_victims = np.random.randint(7, 10)
        for _ in range(num_victims):
            x = np.random.uniform(self.search_area[0], self.search_area[1])
            y = np.random.uniform(self.search_area[2], self.search_area[3])
            
            # Create a visual marker for the victim
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE, 
                radius=victim_sphere_radius, 
                rgbaColor=[1, 0, 0, 0.7]  # Semi-transparent red
            )
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE, 
                radius=victim_sphere_radius
            )
            victim_id = p.createMultiBody(
                baseMass=0,  # Static object
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[x/5, y/5, 0]
            )
            
            self.victims.append({
                'pos': [x/5, y/5, 0], 
                'found': False, 
                'body_id': victim_id
            })
        
        # Reset tracking
        self.visited_cells = set()
        self.current_step = 0
        
        return self._get_obs()

    def _update_fov_visualization(self):
        # Remove previous FOV visualization
        for line in self.fov_lines:
            p.removeUserDebugItem(line)
        self.fov_lines.clear()
        
        # Get current drone position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        
        # Create FOV circle visualization
        num_segments = 12
        points = []
        
        # Calculate points for FOV circle
        for i in range(num_segments + 1):
            angle = 2 * math.pi * i / num_segments
            x = pos[0] + self.fov_radius * math.cos(angle)
            y = pos[1] + self.fov_radius * math.sin(angle)
            points.append([x, y])
        
        # Draw FOV circle segments
        color = [0, 1, 1, 0.5]  # Cyan color with 0.5 alpha
        for i in range(num_segments):
            line_id = p.addUserDebugLine(
                [points[i][0], points[i][1], 0.1],  # Slightly above ground
                [points[i+1][0], points[i+1][1], 0.1],
                color,
                lineWidth=2.0
            )
            self.fov_lines.append(line_id)
        
        # Add lines from drone to circle edge to show FOV angle
        fov_angle_rad = math.radians(self.fov_angle)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        # Left and right FOV lines
        for angle_offset in [-fov_angle_rad/2, fov_angle_rad/2]:
            angle = yaw + angle_offset
            end_x = pos[0] + self.fov_radius * math.cos(angle)
            end_y = pos[1] + self.fov_radius * math.sin(angle)
            line_id = p.addUserDebugLine(
                [pos[0], pos[1], 0.1],
                [end_x, end_y, 0.1],
                [1, 1, 0, 0.5],  # Yellow color
                lineWidth=2.0
            )
            self.fov_lines.append(line_id)

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()
        # Update FOV visualization
        self._update_fov_visualization()
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps
        self.current_step += 1
        
        return obs, reward, done, {}

    def _apply_action(self, action):
        # Simplified drone control using direct velocity commands
        roll, pitch, yaw = action

        # Set vertical velocity to maintain fixed height
        pos = p.getBasePositionAndOrientation(self.drone)[0]
        height_error = self.fixed_height - pos[2]
        vertical_vel = height_error * 10.0  # P controller for height

        linear_vel = [pitch*50, roll*50, vertical_vel]
        angular_vel = [0, 0, yaw*20]
        p.resetBaseVelocity(self.drone, linear_vel, angular_vel)

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        _, ang_vel = p.getBaseVelocity(self.drone)
        
        # Update coverage map
        grid_x = int((pos[0] - self.search_area[0]) / self.grid_size)
        grid_y = int((pos[1] - self.search_area[2]) / self.grid_size)
        self.visited_cells.add((grid_x, grid_y))
        
        # Create coverage map
        grid_size = self._get_grid_size()
        coverage_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        for (x, y) in self.visited_cells:
            if 0 <= x < grid_size and 0 <= y < grid_size:
                coverage_map[x, y] = 1.0
        
        return {
            'position': np.array(pos, dtype=np.float32),
            'imu': np.concatenate([orn, ang_vel]).astype(np.float32),
            'coverage_map': coverage_map
        }

    def _calculate_reward(self):
        reward = 0.0
        pos = p.getBasePositionAndOrientation(self.drone)[0]

        # Coverage reward
        reward += len(self.visited_cells) * 0.1
        
        # Victim detection with color change
        for victim in self.victims:
            if not victim['found']:
                dx = victim['pos'][0] - pos[0]
                dy = victim['pos'][1] - pos[1]
                distance = np.hypot(dx, dy)
                
                if distance < self.fov_radius:
                    reward += 10.0
                    victim['found'] = True
                    
                    # Change victim color to green when found
                    p.changeVisualShape(
                        victim['body_id'], 
                        -1, 
                        rgbaColor=[0, 1, 0, 0.7]  # Semi-transparent green
                    )
        
        # Stability penalty
        euler = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.drone)[1])
        reward -= 0.1 * (abs(euler[0]) + abs(euler[1]))
        
        # Altitude maintenance
        reward -= 0.1 * abs(pos[2] - 10.0)
        
        return reward

    def close(self):
        p.disconnect()

# Training configuration
config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 900_000,
    "policy_kwargs": {
        "features_extractor_class": CustomFeatureExtractor,
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
    },
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "n_steps": 2048,
    "log_dir": "./drone_logs",
    "save_dir": "./drone_models"
}

def train():
    os.makedirs(config["save_dir"], exist_ok=True)
    env = DummyVecEnv([lambda: SearchDroneEnv()])
    
    model = PPO(
        config["policy_type"],
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=config["log_dir"]
    )
    
    checkpoint = CheckpointCallback(
        save_freq=100_000,
        save_path=config["save_dir"],
        name_prefix="search_drone"
    )
    
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint,
        progress_bar=True
    )
    
    model.save(os.path.join(config["save_dir"], "final_model"))
    env.close()

if __name__ == "__main__":
    train()