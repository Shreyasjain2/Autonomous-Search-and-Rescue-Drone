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
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import time

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
import numpy as np

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
import numpy as np


email_config = {
    'sender_email': 'abhisheksaraff18@gmail.com',
    'sender_password': 'wwtx zfew vgzq odzx',  # For Gmail, use App Password
    'recipient_email': 'abhisheksaraff.cy22@rvce.edu.in',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'use_tls': True,
    'interval': 30 # Send email every 5 minutes
}


        
class MissionMonitor:
    def __init__(self, search_area, grid_size):
        self.search_area = search_area
        self.grid_size = grid_size
        self.grid_dims = (
            int((search_area[1] - search_area[0]) / grid_size),
            int((search_area[3] - search_area[2]) / grid_size)
        )
        self.coverage_map = np.zeros(self.grid_dims)
        self.visitation_heatmap = np.zeros(self.grid_dims)
        self.detection_heatmap = np.zeros(self.grid_dims)
        self.humans_found = []
        self.start_time = time.time()
        
    def update_coverage(self, drone_positions):
        for pos in drone_positions:
            grid_x, grid_y = self._world_to_grid(pos[0], pos[1])
            if 0 <= grid_x < self.grid_dims[0] and 0 <= grid_y < self.grid_dims[1]:
                self.coverage_map[grid_x, grid_y] = 1
                self.visitation_heatmap[grid_x, grid_y] += 1

    def add_human_detection(self, position, confidence=1.0):
        grid_x, grid_y = self._world_to_grid(position[0], position[1])
        self.humans_found.append({
            'position': position,
            'time_found': datetime.now(),
            'grid_position': (grid_x, grid_y)
        })
        if 0 <= grid_x < self.grid_dims[0] and 0 <= grid_y < self.grid_dims[1]:
            self.detection_heatmap[grid_x, grid_y] = confidence

    def _world_to_grid(self, x, y):
        grid_x = int((x - self.search_area[0]) / self.grid_size)
        grid_y = int((y - self.search_area[2]) / self.grid_size)
        return grid_x, grid_y

    def print_mission_status(self):
        elapsed_time = time.time() - self.start_time
        coverage_percentage = (np.sum(self.coverage_map > 0) / self.coverage_map.size) * 100
        
        print("\n=== Mission Status Update ===")
        print(f"Time Elapsed: {elapsed_time:.1f} seconds")
        print(f"Area Coverage: {coverage_percentage:.1f}%")
        print(f"Humans Found: {len(self.humans_found)}")
        print("Human Locations:")
        for i, human in enumerate(self.humans_found, 1):
            print(f"  {i}. Position: ({human['position'][0]:.1f}, {human['position'][1]:.1f})")
            print(f"     Found at: {human['time_found']}")
        
        # Calculate and suggest rescue routes
        if self.humans_found:
            print("\nSuggested Rescue Routes:")
            current_pos = (0, 0)  # Assume rescue team starts from center
            unvisited = self.humans_found.copy()
            route = []
            
            while unvisited:
                next_target = min(unvisited, 
                    key=lambda h: np.hypot(
                        h['position'][0] - current_pos[0],
                        h['position'][1] - current_pos[1]
                    ))
                route.append(next_target)
                current_pos = next_target['position']
                unvisited.remove(next_target)
            
            for i, target in enumerate(route, 1):
                print(f"  {i}. Go to ({target['position'][0]:.1f}, {target['position'][1]:.1f})")


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
import numpy as np

class EmailMissionMonitor(MissionMonitor):
    def __init__(self, search_area, grid_size, email_config):
        super().__init__(search_area, grid_size)
        self.email_config = email_config
        self.last_email_time = time.time()
        self.email_interval = email_config.get('interval', 20)  # Default 5 minutes

    def _calculate_mission_stats(self):
        elapsed_time = time.time() - self.start_time
        coverage_percentage = (np.sum(self.coverage_map > 0) / self.coverage_map.size) * 100
        
        # Calculate search efficiency
        search_speed = coverage_percentage / (elapsed_time / 60) if elapsed_time > 0 else 0
        
        # Calculate search patterns
        visited_areas = np.count_nonzero(self.visitation_heatmap)
        revisit_ratio = (np.sum(self.visitation_heatmap > 1) / visited_areas) if visited_areas > 0 else 0
        
        return {
            'elapsed_time': elapsed_time,
            'coverage_percentage': coverage_percentage,
            'humans_found': len(self.humans_found),
            'search_speed': search_speed,
            'revisit_ratio': revisit_ratio * 100,
            'total_area': abs((self.search_area[1] - self.search_area[0]) * (self.search_area[3] - self.search_area[2])),
            'covered_area': (coverage_percentage/100) * abs((self.search_area[1] - self.search_area[0]) * (self.search_area[3] - self.search_area[2]))
        }
        
    def send_status_email(self, force=False):
        current_time = time.time()
        
        if not force and (current_time - self.last_email_time) < self.email_interval:
            return
            
        stats = self._calculate_mission_stats()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #2c3e50;
                    background-color: #f4f7f9;
                    padding: 0 20px;
                    max-width: 900px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #0077cc;
                    color: white;
                    padding: 25px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .header h1 {{
                    margin: 0 0 10px;
                    font-size: 1.8em;
                }}
                .header p {{
                    font-size: 0.9em;
                    margin: 0;
                }}
                .content {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 25px;
                    margin: 30px 0;
                }}
                .stat-box {{
                    background-color: #f9fbfd;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }}
                .stat-box:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }}
                .stat-box h3 {{
                    margin-bottom: 10px;
                    font-size: 1.1em;
                    color: #0077cc;
                }}
                .stat-box p {{
                    font-size: 1em;
                    margin: 0;
                }}
                .human-card {{
                    background-color: #ffffff;
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid #0077cc;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }}
                .human-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }}
                .human-card h3 {{
                    font-size: 1.2em;
                    margin-bottom: 5px;
                    color: #0077cc;
                }}
                .priority {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .success {{
                    color: #27ae60;
                }}
                .warning {{
                    color: #e67e22;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚁 Search and Rescue Mission Status</h1>
                <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <div class="stats-grid">
                    <div class="stat-box">
                        <h3>Mission Duration</h3>
                        <p>{int(stats['elapsed_time']//3600)}h {int((stats['elapsed_time']%3600)//60)}m {int(stats['elapsed_time']%60)}s</p>
                    </div>
                    <div class="stat-box">
                        <h3>Humans Located</h3>
                        <p class="{('warning' if stats['humans_found'] == 0 else 'success')}">
                            {stats['humans_found']} individuals
                        </p>
                    </div>
                    <div class="stat-box">
                        <h3>Area Coverage</h3>
                        <p>{stats['coverage_percentage']:.1f}% ({stats['covered_area']:.1f} sq units)</p>
                        <p class="{'warning' if stats['coverage_percentage'] < 50 else 'success'}">
                            Search Speed: {stats['search_speed']:.2f}% per minute
                        </p>
                    </div>
                    <div class="stat-box">
                        <h3>Search Efficiency</h3>
                        <p>Area Revisit Rate: {stats['revisit_ratio']:.1f}%</p>
                    </div>
                </div>

                <h2>Located Individuals</h2>
        """
        
        if self.humans_found:
            for i, human in enumerate(self.humans_found, 1):
                html_content += f"""
                <div class="human-card">
                    <h3>Individual #{i}</h3>
                    <p><strong>Location:</strong> ({human['position'][0]:.1f}, {human['position'][1]:.1f})</p>
                    <p><strong>Found at:</strong> {human['time_found']}</p>
                    <p><strong>Time since discovery:</strong> {(datetime.now() - human['time_found']).total_seconds()//60:.0f} minutes</p>
                </div>
                """

        else:
            html_content += """
            <div class="human-card warning">
                <p>No individuals located yet. Search continuing...</p>
            </div>
            """

        if self.humans_found:
            html_content += """
            <div class="rescue-route">
                <h2>📍 Optimized Rescue Route</h2>
            """
            
            current_pos = (0, 0)
            unvisited = self.humans_found.copy()
            route = []
            
            while unvisited:
                next_target = min(unvisited, 
                    key=lambda h: np.hypot(
                        h['position'][0] - current_pos[0],
                        h['position'][1] - current_pos[1]
                    ))
                route.append(next_target)
                current_pos = next_target['position']
                unvisited.remove(next_target)
            
            for i, target in enumerate(route, 1):
                html_content += f"""
                <p>{i}. Navigate to coordinates ({target['position'][0]:.1f}, {target['position'][1]:.1f})</p>
                """
                
            html_content += "</div>"

        html_content += """
            </div>
        </body>
        </html>
        """

        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'🚨 Search and Rescue Status - {len(self.humans_found)} Located'
        msg['From'] = self.email_config['sender_email']
        msg['To'] = self.email_config['recipient_email']
        
        # Add both plain text and HTML versions
        text_content = f"Mission Status Update\nTime: {datetime.now()}\nHumans Found: {len(self.humans_found)}\nCoverage: {stats['coverage_percentage']:.1f}%"
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        try:
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                server.login(
                    self.email_config['sender_email'],
                    self.email_config['sender_password']
                )
                server.send_message(msg)
            
            self.last_email_time = current_time
            print(f"Status email sent to {self.email_config['recipient_email']}")
            
        except Exception as e:
            print(f"Failed to send status email: {str(e)}")

    def print_mission_status(self):
        super().print_mission_status()
        self.send_status_email()

class DroneSwarm:
    def __init__(self, num_drones, search_area, monitor):
        self.num_drones = num_drones
        self.search_area = search_area
        self.monitor = monitor
        self.drones = []
        self.shared_memory = {
            'visited_areas': set(),
            'detected_humans': set(),
            'drone_positions': defaultdict(lambda: (0, 0, 0))
        }

    def assign_search_areas(self):
        # Calculate the dimensions of the total search area
        width = self.search_area[1] - self.search_area[0]
        height = self.search_area[3] - self.search_area[2]
        
        num_cols = int(np.ceil(np.sqrt(self.num_drones)))
        num_rows = int(np.ceil(self.num_drones / num_cols))
        
        cell_width = width / num_cols
        cell_height = height / num_rows
        
        assignments = []
        # print("\nCalculating search area assignments:")
        for i in range(self.num_drones):
            row = i // num_cols
            col = i % num_cols
            
            x_min = self.search_area[0] + col * cell_width
            x_max = x_min + cell_width
            y_min = self.search_area[2] + row * cell_height
            y_max = y_min + cell_height
            
            # print(f"Drone {i+1} area: x:[{x_min:.1f}, {x_max:.1f}], y:[{y_min:.1f}, {y_max:.1f}]")
            assignments.append([x_min, x_max, y_min, y_max])
        
        return assignments

    def update_shared_memory(self, drone_id, position, detected_humans):
        self.shared_memory['drone_positions'][drone_id] = position
        grid_pos = self.monitor._world_to_grid(position[0], position[1])
        self.shared_memory['visited_areas'].add(grid_pos)
        
        for human in detected_humans:
            if tuple(human) not in self.shared_memory['detected_humans']:
                self.shared_memory['detected_humans'].add(tuple(human))
                self.monitor.add_human_detection(human)

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=512)
        
        self.position_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.imu_net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Dynamically calculate convolutional output size
        coverage_shape = observation_space["coverage_map"].shape
        h, w = coverage_shape
        
        # First convolution keeps dimensions due to padding=1
        # Second convolution: (H-3)//2 +1, (W-3)//2 +1
        conv2_h = (h - 3) // 2 + 1
        conv2_w = (w - 3) // 2 + 1
        self.conv_output_size = 32 * conv2_h * conv2_w
        
        self.coverage_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 256)
        )
        
        self.combine_net = nn.Sequential(
            nn.Linear(128 + 128 + 256, 512),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        position_features = self.position_net(observations["position"])
        imu_features = self.imu_net(observations["imu"])
        
        maps = torch.cat([
            observations["coverage_map"].unsqueeze(1),
            observations["visitation_map"].unsqueeze(1)
        ], dim=1)
        
        coverage_features = self.coverage_net(maps)
        combined = torch.cat([position_features, imu_features, coverage_features], dim=1)
        return self.combine_net(combined)

class MultiDroneEnv(gym.Env):
    def __init__(self, num_drones=4, render=True):
        super(MultiDroneEnv, self).__init__()
        self.render_mode = render
        self.search_area = [-15, 15, -15, 15]
        self.grid_size = 2
        self.num_drones = num_drones
        
        # Initialize mission monitor
        # In MultiDroneEnv.__init__
        self.monitor = EmailMissionMonitor(self.search_area, self.grid_size, email_config)
        self.swarm = DroneSwarm(num_drones, self.search_area, self.monitor)
        
        # PyBullet setup
        self.client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Environment parameters
        self.fov_radius = 2
        self.fov_angle = 120
        self.max_steps = 50000
        self.fixed_height = 2.0
        self.status_update_interval = 500  # Steps between status updates
        
        # Action and Observation Spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1] * num_drones),
            high=np.array([1, 1, 1] * num_drones),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'imu': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'coverage_map': spaces.Box(low=0, high=1, 
                shape=(self._get_grid_size(), self._get_grid_size()), dtype=np.float32),
            'visitation_map': spaces.Box(low=0, high=np.inf, 
                shape=(self._get_grid_size(), self._get_grid_size()), dtype=np.float32)
        })
        
        self.drones = []
        self.victims = []
        self.current_step = 0
        self.last_status_update = 0

    def _get_grid_size(self):
        # Add error checking for grid size calculation
        width = self.search_area[1] - self.search_area[0]
        height = self.search_area[3] - self.search_area[2]
        grid_width = int(width / self.grid_size)
        grid_height = int(height / self.grid_size)
        
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError(f"Invalid grid dimensions: {grid_width}x{grid_height}. Check search_area and grid_size parameters.")
        
        # print(f"Grid dimensions: {grid_width}x{grid_height}")
        return grid_width  # Assuming square grid

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load environment
        self._setup_environment()
        
        # Initialize drones with position checking
        self.drones = []
        search_areas = self.swarm.assign_search_areas()
        
        # print("\nInitializing drones:")
        for i in range(self.num_drones):
            area = search_areas[i]
            start_pos = [
                np.clip((area[0] + area[1]) / 2, self.search_area[0], self.search_area[1]),
                np.clip((area[2] + area[3]) / 2, self.search_area[2], self.search_area[3]),
                self.fixed_height
            ]
            
            # print(f"Drone {i+1}: Assigned area: {area}")
            # print(f"Drone {i+1}: Start position: {start_pos}")
            
            try:
                drone = p.loadURDF("./data/Quadrotor/quadrotor.urdf", start_pos, globalScaling=2.0)
                self.drones.append(drone)
            except Exception as e:
                print(f"Error initializing drone {i+1}: {str(e)}")
                raise
        
        # Generate and visualize victims
        self._generate_victims()
        
        self.current_step = 0
        self.last_status_update = 0
        
        return self._get_obs()

    def _setup_environment(self):
        plane = p.loadURDF("./data/plane.urdf", useMaximalCoordinates=True)
        p.changeVisualShape(plane, -1, rgbaColor=[0.5, 0.4, 0.3, 1])
        
        # Add boundary visualization
        for i in range(4):
            start = [
                self.search_area[0] if i < 2 else self.search_area[1],
                self.search_area[2] if i % 2 == 0 else self.search_area[3],
                0
            ]
            end = [
                self.search_area[1] if i < 2 else self.search_area[0],
                self.search_area[2] if i % 2 == 0 else self.search_area[3],
                0
            ]
            p.addUserDebugLine(start, end, [1, 0, 0], 2)

    def _generate_victims(self):
        self.victims = []
        num_victims = np.random.randint(7, 15)
        
        # Add debug print
        # print(f"\nGenerating {num_victims} victims in search area: {self.search_area}")
        
        for i in range(num_victims):
            x = np.random.uniform(self.search_area[0], self.search_area[1])
            y = np.random.uniform(self.search_area[2], self.search_area[3])
            
            # Add debug print
            # print(f"Victim {i+1}: Position before scaling: ({x}, {y})")
            
            # Remove the division by 5 that was in the original code
            # as it was incorrectly scaling the positions
            victim_pos = [x, y, 0]
            
            # print(f"Victim {i+1}: Final position: ({victim_pos[0]}, {victim_pos[1]})")
            
            try:
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.5,
                    rgbaColor=[1, 0, 0, 0.7]
                )
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.5
                )
                victim_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=victim_pos
                )
                
                self.victims.append({
                    'pos': victim_pos,
                    'found': False,
                    'body_id': victim_id
                })
            except Exception as e:
                print(f"Error creating victim {i+1}: {str(e)}")
                raise


    def step(self, actions):
        # Reshape actions for multiple drones
        actions = actions.reshape(self.num_drones, 3)
        
        # Apply actions to each drone
        for i, drone in enumerate(self.drones):
            self._apply_action(drone, actions[i])
        
        p.stepSimulation()
        
        # Get observations and calculate rewards
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps
        
        # Update mission monitor
        drone_positions = [p.getBasePositionAndOrientation(drone)[0] for drone in self.drones]
        self.monitor.update_coverage(drone_positions)
        
        # Print status update if interval has passed
        if self.current_step - self.last_status_update >= self.status_update_interval:
            self.monitor.print_mission_status()
            self.last_status_update = self.current_step
        
        self.current_step += 1
        
        return obs, reward, done, {}

    def _apply_action(self, drone, action):
        roll, pitch, yaw = action
        pos = p.getBasePositionAndOrientation(drone)[0]
        height_error = self.fixed_height - pos[2]
        vertical_vel = height_error * 10.0
        
        linear_vel = [pitch*50, roll*50, vertical_vel]
        angular_vel = [0, 0, yaw*20]
        p.resetBaseVelocity(drone, linear_vel, angular_vel)

    def _get_obs(self):
        # Get all drone positions and orientations
        drone_states = [p.getBasePositionAndOrientation(drone) for drone in self.drones]
        positions = [state[0] for state in drone_states]
        orientations = [state[1] for state in drone_states]
        ang_vels = [p.getBaseVelocity(drone)[1] for drone in self.drones]
        
        # Update shared memory
        for i, pos in enumerate(positions):
            detected_humans = self._check_human_detection(pos)
            self.swarm.update_shared_memory(i, pos, detected_humans)
        
        # Create combined observation
        grid_size = self._get_grid_size()
        coverage_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        visitation_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Update maps based on all drone positions
        for pos in positions:
            grid_x, grid_y = self._world_to_grid(pos[0], pos[1])
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                coverage_map[grid_x, grid_y] = 1
                visitation_map[grid_x, grid_y] += 1
        
        # Return observation for the lead drone (others will use this with slight modifications)
        return {
            'position': np.array(positions[0], dtype=np.float32),
            'imu': np.concatenate([orientations[0], ang_vels[0]]).astype(np.float32),
            'coverage_map': coverage_map,
            'visitation_map': visitation_map
        }

    def _check_human_detection(self, drone_pos):
        detected = []
        for victim in self.victims:
            if not victim['found']:
                dx = victim['pos'][0] - drone_pos[0]
                dy = victim['pos'][1] - drone_pos[1]
                distance = np.hypot(dx, dy)
                
                if distance < self.fov_radius:
                    victim['found'] = True
                    detected.append(victim['pos'])
                    # Change victim color to green when found
                    p.changeVisualShape(
                        victim['body_id'],
                        -1,
                        rgbaColor=[0, 1, 0, 0.7]
                    )
        return detected

    def _world_to_grid(self, x, y):
        grid_x = int((x - self.search_area[0]) / self.grid_size)
        grid_y = int((y - self.search_area[2]) / self.grid_size)
        return grid_x, grid_y

    def _calculate_reward(self):
        total_reward = 0.0
        
        # Get all drone positions
        drone_positions = [p.getBasePositionAndOrientation(drone)[0] for drone in self.drones]
        
        for pos in drone_positions:
            # Boundary penalty
            if (pos[0] < self.search_area[0] or pos[0] > self.search_area[1] or
                pos[1] < self.search_area[2] or pos[1] > self.search_area[3]):
                total_reward -= 5.0
            
            # Height maintenance reward
            total_reward -= 0.1 * abs(pos[2] - self.fixed_height)
            
            # Stability reward
            orientation = p.getBasePositionAndOrientation(self.drones[0])[1]
            euler = p.getEulerFromQuaternion(orientation)
            total_reward -= 0.1 * (abs(euler[0]) + abs(euler[1]))
        
        # Coverage reward (based on newly covered areas)
        current_coverage = len(self.swarm.shared_memory['visited_areas'])
        total_reward += 0.5 * current_coverage
        
        # Human detection reward
        total_reward += 10.0 * len(self.swarm.shared_memory['detected_humans'])
        
        # Swarm cohesion reward
        cohesion_reward = self._calculate_swarm_cohesion(drone_positions)
        total_reward += cohesion_reward
        
        return total_reward

    def _calculate_swarm_cohesion(self, drone_positions):
        if len(drone_positions) < 2:
            return 0.0
        
        # Calculate average distance between drones
        distances = []
        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                dist = np.hypot(
                    drone_positions[i][0] - drone_positions[j][0],
                    drone_positions[i][1] - drone_positions[j][1]
                )
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        optimal_distance = 20.0  # Desired distance between drones
        
        # Penalize if drones are too close or too far apart
        return -0.1 * abs(avg_distance - optimal_distance)

    def close(self):
        p.disconnect()

def train_multi_drone():
    # Create directories for logs and models
    os.makedirs("./drone_logs", exist_ok=True)
    os.makedirs("./drone_models", exist_ok=True)
    
    # Initialize environment
    env = DummyVecEnv([lambda: MultiDroneEnv(num_drones=4)])
    
    # Initialize model with custom policy
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        policy_kwargs={
            "features_extractor_class": CustomFeatureExtractor,
            "net_arch": [dict(pi=[512, 512], vf=[512, 512])]
        },
        verbose=1,
        tensorboard_log="./drone_logs"
    )
    
    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./drone_models",
        name_prefix="multi_drone"
    )

    
    
    # Train the model
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("./drone_models/final_multi_drone_model")
    env.close()

def test_multi_drone(model_path="./drone_models/final_multi_drone_model"):
    env = MultiDroneEnv(num_drones=4, render=True)
    model = PPO.load(model_path)
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        time.sleep(0.01)  # Slow down visualization
    
    env.close()

if __name__ == "__main__":

    print("Starting multi-drone search and rescue training...")
    train_multi_drone()
    
    print("\nTraining completed. Starting test simulation...")
    test_multi_drone()