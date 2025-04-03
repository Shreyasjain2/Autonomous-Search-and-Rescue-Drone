# 🛁 Autonomous Search and Rescue Drone System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/simulation-PyBullet-red)](https://pybullet.org)
[![Stable Baselines3](https://img.shields.io/badge/RL-StableBaselines3-orange)](https://stable-baselines3.readthedocs.io/)

## 🗃️ Overview

This project implements an autonomous multi-drone system for search and rescue operations using reinforcement learning (RL). The system trains drone swarms to efficiently search areas, detect victims, and optimize rescue routes while maintaining stable flight patterns. The project consists of:

![Multi Drone Training](https://github.com/Prakharjain1509/Autonomous-Search-and-Rescue-Drone/blob/e1a800a7aed0cd0b18775f1ce3a5e279343b5a83/assets/multi.png)

- A **real-time drone UI** for monitoring and controlling drones
- **Reinforcement learning** models to optimize drone search efficiency
- **Simulation-based training** with PyBullet for drone behavior learning
- **Autonomous victim detection** and rescue path optimization

## 🔍 Key Features

- **Autonomous Drone Operations**: Drones navigate and search independently
- **Multi-sensor Integration**: Uses position, IMU, and FOV data
- **Victim Detection & Rescue Planning**: AI-driven decision-making for optimal rescue routes
- **Real-time UI Dashboard**: Monitor drone behavior and RL training
- **Reinforcement Learning**: PPO-based training for optimized navigation

![RL Dashboard](https://github.com/Prakharjain1509/Autonomous-Search-and-Rescue-Drone/blob/e1a800a7aed0cd0b18775f1ce3a5e279343b5a83/assets/frontend.png)

## 👩‍💻 System Components

### 1. **Drone UI**
A React and Vite-powered dashboard for real-time visualization and control.

### 2. **Backend (Drone-UI)**
A Python-based backend for processing drone input and managing communication.

### 3. **Reinforcement Learning Module (Drone-RL-Processing)**
Trains drones using PPO with PyBullet simulation.

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Prakharjain1509/Autonomous-Search-and-Rescue-Drone.git
cd Autonomous-Search-and-Rescue-Drone
```

### 2. Backend Setup (Drone-UI)
```bash
cd Drone-UI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backend.py
```

### 3. Frontend Setup (Drone-Dashboard-Frontend)
```bash
cd ../Drone-Dashboard-Frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to view the UI.

### 4. Running the Basic UI (Static Page)
To serve the basic UI from `Drone-UI/index.html`:
```bash
cd ../Drone-UI
python -m http.server
```
Navigate to `http://localhost:8000/index.html` to view the UI.

---
![Basic UI](https://github.com/Prakharjain1509/Autonomous-Search-and-Rescue-Drone/blob/main/Drone-RL-Processing/assets/frontend_drone.png)

## 🚀 Running the RL Simulation

### 1. Setup the RL Environment
```bash
cd ../Drone-RL-Processing
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start RL Training & Simulation
```bash
# Run the single drone simulation with FOV visualization
python singleDroneWithFOV.py

# Optimized single drone simulation
python singleDrone_final.py

# Multi-drone simulation
python fourDrones_final.py
```

### 3. Start RL Dashboard
```bash
cd RL_Training_FrontendReport
npm install
npm start
```
Navigate to `http://localhost:3000` to view training metrics.

---

## 📊 Performance Metrics

- **Search Efficiency**: Evaluates coverage per unit time
- **Victim Detection Rate**: Measures success in locating victims
- **Revisit Rate**: Tracks unnecessary redundant searches
- **Flight Stability**: Ensures stable drone movement during searches

## 💌 Email Notification System

Mission progress updates are automatically sent via email:
- Search coverage statistics
- Victim detection details
- Optimized rescue paths

---

## 📖 License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

