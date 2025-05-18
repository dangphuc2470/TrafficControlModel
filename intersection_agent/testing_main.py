from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
import configparser
import socket
import argparse
import sys

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path, get_latest_model_for_agent


def read_server_config(config_file='server_config_1.ini'):
    if not os.path.exists(config_file):
        return None, None, None, None
    config = configparser.ConfigParser()
    config.read(config_file)
    if 'server' not in config:
        return None, None, None, None
    if not config['server'].getboolean('enabled', fallback=False):
        return None, None, None, None
    server_url = config['server'].get('server_url', None)
    agent_id = config['server'].get('agent_id', socket.gethostname())
    # Read location data if available
    location_data = None
    if 'location' in config:
        location_data = {
            'latitude': config['location'].get('latitude', None),
            'longitude': config['location'].get('longitude', None),
            'intersection_name': config['location'].get('intersection_name', f'Intersection {agent_id}'),
            'orientation': config['location'].get('orientation', '0')
        }
    # Read map configuration
    map_config = {}
    if 'map' in config:
        map_config = {
            'send_topology': config['map'].getboolean('send_topology', True),
            'environment_file': config['map'].get('environment_file', 'intersection/environment.net.xml'),
            'connection_distance': config['map'].getfloat('connection_distance', 1.5),
            'connected_to': [x.strip() for x in config['map'].get('connected_to', '').split(',') if x.strip()]
        }
    else:
        map_config = {
            'send_topology': True,
            'environment_file': 'intersection/environment.net.xml',
            'connection_distance': 1.5,
            'connected_to': []
        }
    # Read visualization options
    viz_config = {}
    if 'visualization' in config:
        viz_config = {
            'marker_color': config['visualization'].get('marker_color', 'green'),
            'marker_icon': config['visualization'].get('marker_icon', 'traffic-light')
        }
    # Combine everything into a single mapping config
    mapping_config = {
        'location': location_data,
        'map': map_config,
        'visualization': viz_config
    }
    # Define the environment file path based on map config
    env_file_path = map_config['environment_file'] if map_config['send_topology'] else None
    if env_file_path and not os.path.exists(env_file_path):
        print(f"Warning: Environment file not found at {env_file_path}")
        env_file_path = None
    return server_url, agent_id, mapping_config, env_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-config', type=str, default='server_config_1.ini')
    args = parser.parse_args()

    config = import_test_configuration(config_file='testing_settings.ini')
    
    # Read server configuration first to get agent_id
    server_url, agent_id, mapping_config, env_file_path = read_server_config(args.server_config)
    
    # Extract agent number from agent_id (e.g., "agent1" -> "1")
    agent_number = agent_id.replace('agent', '') if agent_id.startswith('agent') else agent_id
    
    # Get the latest model for this specific agent
    latest_model = get_latest_model_for_agent(config['models_path_name'], agent_number)
    if latest_model is None:
        sys.exit(f"No models found for agent {agent_number}")
    
    print(f"\nUsing latest model for agent {agent_number}: model_{latest_model}")
    config['model_to_test'] = latest_model
    
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    if server_url:
        print(f"Connecting to central server at {server_url} as agent {agent_id}")
    else:
        print("Running in standalone mode (no central server)")

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )

    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        server_url,
        agent_id,
        mapping_config,
        env_file_path
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
