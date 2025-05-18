# Traffic Light Control System using Deep Reinforcement Learning

This project implements a distributed traffic light control system where multiple intersection agents learn optimal traffic signal timing through Deep Q-Learning. The system includes a central server for coordination and visualization.

## Origin 
This project is adapted from the work by Andrea Vidali on [Deep Q-Learning for Traffic Signal Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control). The original repository provides a foundation for applying reinforcement learning to traffic management problems, which we've extended with multi-agent coordination capabilities and a central server architecture.


## Features

* **Deep Q-Network (DQN) learning** for traffic signal control.
* **Central server** for traffic coordination between intersections.
* **Real-time visualization dashboard** with performance metrics.
* **Interactive network map** showing intersection status.
* **Configurable simulation parameters.**
* **Green wave coordination** based on geographical distance and travel time.

## Requirements

* Python 3.7.0
* SUMO Traffic Simulator 1.8.0+ ([official guide](https://sumo.dlr.de/docs/Installation.html))
* TensorFlow 2.0.0
* TensorFLow GPU
* Flask
* NumPy
* Matplotlib
* Folium (for maps)
* ... Check requirement.txt for all lib I have installed

## Project Structure


The project directory is organized as follows:

- `intersection_agent/`: Contains the RL agent implementation.
- `central_server/`: Central server for coordination and visualization.

*(Dashboard-related files are listed below)*
- `static/`: Static assets for the web interface.
- `templates/`: HTML templates for the dashboard.
- `...`: Other project-level files (e.g., configuration examples, documentation).



## Installation

1.  Install SUMO Traffic Simulator following the [official guide](https://sumo.dlr.de/docs/Installation.html).
2.  Clone the repository:

    ```bash
    git clone [https://github.com/yourusername/TrafficControlModel.git](https://github.com/yourusername/TrafficControlModel.git)
    cd TrafficControlModel
    ```

3.  Install the required Python packages:

    ```bash
    pip install tensorflow flask numpy matplotlib folium
    ```

## Running the System
Note that I am using **Python 3.13.3** for **central server** and **Python 3.8.20** for Intersection and Sync agent
1.  **Start the Central Server**

    ```bash
    python ./central_server/central_server.py
    ```

    The central server will start on `http://localhost:5000`.

2.  **Run Agent Training**

    To start training with a specific configuration:

    ```bash
    cd /intersection_agent

    python3.8 train_with_server.py --server-config server_config_1.ini
    ```

    You can run multiple agents with different configurations by using different config files:

    ```bash
    python3.8 train_with_server.py --server-config server_config_2.ini
    ```

3.  **View the Dashboard**

    Open your browser and navigate to:

    * **Dashboard:** `http://localhost:5000/`
    * **Network Map:** `http://localhost:5000/map`

## Configuration Files

* `server_config_*.ini`: Defines agent-specific settings including:
    * Agent ID and location
    * Server URL
    * Training parameters (learning rate, episodes, etc.)
    * Traffic patterns

## Monitoring Performance

During training, you can monitor:

* Average queue length
* Cumulative waiting time
* Reward values
* Traffic flow visualization

The central server dashboard provides real-time updates on agent performance and network-wide traffic patterns.


## Testing

### Running Tests with Server

To test an intersection agent with the central server:

```bash
# Test the base model
python3.8 test_with_server.py --server-config server_config_1.ini --phase base


# Test with sync-aware model
python3.8 test_with_server.py --server-config server_config_1.ini --phase sync
```
or just 
```bash
python3.8 test_with_server.py --server-config server_config_1.ini 
```

The script will:
1. Read the server configuration to get the agent ID
2. Find the latest model for that agent:
   - For sync-aware models (default): `models/model_XXX/intersection_agentY_model.h5`
   - For base models: `models/model_XXX/trained_model_base.h5`
3. Connect to the central server and run the simulation
4. Display results including:
   - Average reward
   - Total reward
   - Average queue length

### Model Types

1. **Sync-aware Models** (Default)
   - Located in: `models/model_XXX/intersection_agentY_model.h5`
   - Used by default (no phase specified)
   - Requires sync_agent to be running
   - Example: `models/model_142/intersection_agent1_model.h5`

2. **Base Models**
   - Located in: `models/model_XXX/trained_model_base.h5`
   - Used when `--phase base` is specified
   - Does not require sync_agent
   - Example: `models/model_96/trained_model_base.h5`

### Running with Sync Agent

To use sync-aware models, you need to run the sync agent:

```bash
# Start the sync agent
python3.8 sync_agent/sync_agent.py --server-config server_config_1.ini

# Then run the intersection agent with sync-aware model
python3.8 test_with_server.py --server-config server_config_1.ini
```

### Server Configuration

The server configuration file (e.g., `server_config_1.ini`) should contain:
```ini
[server]
enabled = true
server_url = http://127.0.0.1:5000
agent_id = agent1

[location]
latitude = 10.123
longitude = 106.456
intersection_name = Intersection 1
orientation = 0

[map]
send_topology = true
environment_file = intersection/environment.net.xml
connection_distance = 1.5
connected_to = agent2,agent3

[visualization]
marker_color = green
marker_icon = traffic-light
```

### Testing Settings

The testing settings are configured in `testing_settings.ini`:
```ini
[simulation]
gui = True
max_steps = 5400
n_cars_generated = 1000
episode_seed = 10000
yellow_duration = 4
green_duration = 10

[agent]
num_states = 80
num_actions = 4

[dir]
models_path_name = models
sumocfg_file_name = sumo_config.sumocfg
model_to_test = 96
```

### Output

The test will display:
```
=== Model Information ===
Agent ID: agent1
Using model: 142
Model path: models/model_142/intersection_agent1_model.h5
Plot path: models/plots_142
Using sync-aware model (requires sync_agent)
=======================

----- Testing episode
Simulating...
Simulation time: 123.4 s

----- Results -----
Average reward: 123.45
Total reward: 1234.56
Average queue length: 5.67

End of testing
```

### Notes
1. Make sure the central server is running before starting the test
2. The agent ID in the server config must match the model files
3. For sync-aware models, ensure the sync_agent is running
4. The test will automatically find the latest model for the specified agent and type
