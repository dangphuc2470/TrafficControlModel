from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot
import os
import sys
import datetime
import numpy as np
import configparser
import socket

#import from outside folder
from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from utils import import_train_configuration, set_sumo, set_train_path
from agent_communicator import AgentCommunicatorTraining


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def read_server_config(config_file='server_config.ini'):
    """Read the server configuration file"""
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
    # Parse command line arguments to specify config file
    import argparse
    parser = argparse.ArgumentParser(description='Train traffic light control agent with server connection')
    parser.add_argument('--server-config', type=str, default='server_config.ini',
                       help='Path to the server configuration file (default: server_config.ini)')
    args = parser.parse_args()
    
    # Load training configuration 
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    # Read server configuration from specified file
    server_url, agent_id, mapping_config, env_file_path = read_server_config(config_file=args.server_config)
    
    if server_url:
        print(f"Connecting to central server at {server_url} as agent {agent_id}")
        print(f"Using server config: {args.server_config}")
        
        # Display mapping configuration info if available
        if mapping_config and 'location' in mapping_config and mapping_config['location']:
            location = mapping_config['location']
            print(f"Location: {location.get('intersection_name')} " +
                  f"({location.get('latitude')}, {location.get('longitude')})")
        
        if env_file_path:
            print(f"Using environment file: {env_file_path}")
    else:
        print("Running in standalone mode (no central server)")

    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        config['num_states'], 
        config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        server_url,
        agent_id,
        mapping_config,
        env_file_path,
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Training finished -----")
    print("Starting time:", timestamp_start)
    print("Ending time:", datetime.datetime.now())
    print("Session info saved at:", path)
    
    Model.save_model(path)
    
    # Final cleanup
    Simulation.cleanup()