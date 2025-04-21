from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import datetime
import numpy as np
import configparser
import socket

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from utils import import_test_configuration, set_sumo, set_test_path
from agent.agent_communicator import AgentCommunicator

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

# Modified Simulation class that integrates with the server
class TestingSimulationWithServer(Simulation):
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, 
                 yellow_duration, num_states, num_actions, server_url=None, agent_id=None):
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, 
                         yellow_duration, num_states, num_actions)
        # Initialize server communication if URL is provided
        self.server_url = server_url
        if server_url:
            self.communicator = AgentCommunicator(server_url, agent_id)
            self.communicator.update_status("test_initialized")
            self.communicator.update_config({
                "max_steps": max_steps,
                "green_duration": green_duration,
                "yellow_duration": yellow_duration,
                "num_states": num_states,
                "num_actions": num_actions,
                "mode": "testing"
            })
            self.communicator.start_background_sync()
        else:
            self.communicator = None
    
    def run(self, episode):
        """
        Runs the testing simulation and reports to server if enabled
        """
        start_time = datetime.datetime.now()
        
        if self.communicator:
            self.communicator.update_status("testing")
            
        # Call the parent class run method
        simulation_time = super().run(episode)
        
        # Report results to server if enabled
        if self.communicator:
            total_reward = np.sum(self._reward_episode)
            avg_queue_length = np.mean(self._queue_length_episode)
            self.communicator.update_episode_result(
                episode=episode,
                reward=total_reward,
                queue_length=avg_queue_length,
                waiting_time=sum(self._queue_length_episode)
            )
            self.communicator.update_status("test_completed")
            
        return simulation_time
    
    def cleanup(self):
        """Clean up when done"""
        if self.communicator:
            self.communicator.update_status("terminated")
            self.communicator.stop_background_sync()
            self.communicator.sync_with_server()  # Final sync

if __name__ == "__main__":
    # Configure the test
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    # Read server configuration
    server_url, agent_id = read_server_config()
    if server_url:
        print(f"Connecting to central server at {server_url} as agent {agent_id}")
    else:
        print("Running in standalone mode (no central server)")

    # Create model and traffic generator
    Model = TestModel(
        config['num_states'],
        model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    # Create the simulation with server integration
    Simulation = TestingSimulationWithServer(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        server_url,
        agent_id
    )
    
    print("----- Testing episode")
    simulation_time = Simulation.run(config['episode_seed'])
    print("Simulation time:", simulation_time, "s")
    
    # Calculate and print statistics
    print("----- Results -----")
    reward_episode = Simulation.reward_episode
    print("Average reward:", np.mean(reward_episode))
    print("Total reward:", np.sum(reward_episode))
    
    queue_length_episode = Simulation.queue_length_episode
    print("Average queue length:", np.mean(queue_length_episode))
    
    print("End of testing")
    
    # Final cleanup
    Simulation.cleanup()