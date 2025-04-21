from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import datetime
import numpy as np
import configparser
import socket
import timeit

# Import from parent directory
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from utils import import_test_configuration, set_sumo, set_test_path
from agent_communicator import AgentCommunicatorTesting

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Import traci only after SUMO_HOME is set
import traci

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

class TestingSimulationWithServer(Simulation):
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, 
                 yellow_duration, num_states, num_actions, server_url=None, agent_id=None):
        # Call the parent constructor
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, 
                         yellow_duration, num_states, num_actions)
        
        # Initialize server communication if URL is provided
        self._server_url = server_url
        self._agent_id = agent_id
        
        if server_url:
            self._communicator = AgentCommunicatorTesting(server_url, agent_id)
            self._communicator.update_status("test_initialized")
            self._communicator.update_config({
                "max_steps": max_steps,
                "green_duration": green_duration,
                "yellow_duration": yellow_duration,
                "num_states": num_states,
                "num_actions": num_actions,
                "mode": "testing"
            })
            self._communicator.start_background_sync()
        else:
            self._communicator = None

    def run(self, episode):
        """
        Runs the testing simulation and reports to server if enabled
        """
        start_time = timeit.default_timer()

        if self._communicator:
            self._communicator.update_status("testing")
            
        # Reset episode arrays
        self._reward_episode = []
        self._queue_length_episode = []
            
        # Generate route file and start simulation
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialize simulation variables
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1  # dummy init

        # Main simulation loop
        while self._step < self._max_steps:
            # Get current state of the intersection
            current_state = self._get_state()

            # Calculate reward of previous action
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Choose the light phase to activate
            action = self._choose_action(current_state)

            # If the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Execute the green phase
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Save variables for next step
            old_action = action
            old_total_wait = current_total_wait

            # Add reward to episode total
            self._reward_episode.append(reward)

            # Update server with intermediate results periodically
            if self._communicator and self._step % 100 == 0:
                self._communicator.update_intermediate_reward(
                    sum(self._reward_episode[-100:]),
                    np.mean(self._queue_length_episode[-100:]) if len(self._queue_length_episode) > 0 else 0
                )

        # End simulation
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        # Report final results to server
        if self._communicator:
            total_reward = np.sum(self._reward_episode)
            avg_queue_length = np.mean(self._queue_length_episode)
            total_waiting_time = np.sum(self._queue_length_episode)
            
            self._communicator.update_episode_result(
                episode=episode,
                reward=total_reward,
                queue_length=avg_queue_length,
                waiting_time=total_waiting_time
            )
            self._communicator.update_status("test_completed")

        return simulation_time

    def cleanup(self):
        """Clean up when done"""
        if self._communicator:
            self._communicator.update_status("test_terminated")
            self._communicator.stop_background_sync()
            self._communicator.sync_with_server()  # Final sync

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