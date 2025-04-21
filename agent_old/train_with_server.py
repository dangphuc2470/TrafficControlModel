from __future__ import absolute_import
from __future__ import print_function

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

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def read_server_config(config_file='server_config.ini'):
    """Read the server configuration file"""
    if not os.path.exists(config_file):
        return None, None
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'server' not in config:
        return None, None
    
    if not config['server'].getboolean('enabled', fallback=False):
        return None, None
    
    server_url = config['server'].get('server_url', None)
    agent_id = config['server'].get('agent_id', socket.gethostname())
    
    return server_url, agent_id

if __name__ == "__main__":
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    # Read server configuration
    server_url, agent_id = read_server_config()
    if server_url:
        print(f"Connecting to central server at {server_url} as agent {agent_id}")
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
        agent_id
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