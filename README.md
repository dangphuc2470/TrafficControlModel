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

- `agent_new/`: Contains the RL agent implementation.
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

1.  **Start the Central Server**

    ```bash
    python ./central_server/central_server.py
    ```

    The central server will start on `http://localhost:5000`.

2.  **Run Agent Training**

    To start training with a specific configuration:

    ```bash
    cd /agent_new

    python train_with_server.py --server-config server_config_1.ini
    ```

    You can run multiple agents with different configurations by using different config files:

    ```bash
    python train_with_server.py --server-config server_config_2.ini
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

## Extending the System

To add new intersections:

1.  Create a new SUMO network configuration.
2.  Add a new server configuration file.
3.  Run the agent with the new configuration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
