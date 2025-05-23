from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import datetime
import numpy as np
import configparser
import socket
import timeit
import traci
import argparse
import glob

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from utils import import_test_configuration, set_sumo, set_test_path
from agent_communicator import AgentCommunicatorTesting
from interactive_simulation import InteractiveSimulation

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def get_latest_model_for_agent(models_dir, agent_id, phase=None):
    """
    Find the latest model for a specific agent
    Args:
        models_dir: Directory containing model folders
        agent_id: Agent ID (e.g., 'agent1')
        phase: If specified, look for phase-based model (e.g., 'base', 'sync')
    Returns:
        tuple: (model_number, model_path) or (None, None) if no model found
    """
    # Extract agent number from agent_id (e.g., 'agent1' -> '1')
    agent_num = agent_id.replace('agent', '')
    
    # Find all model directories
    model_dirs = glob.glob(os.path.join(models_dir, 'model_*'))
    if not model_dirs:
        return None, None
    
    # Sort directories by model number
    model_dirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    
    # Look for the latest model that has a file for this agent
    for model_dir in model_dirs:
        model_num = int(model_dir.split('_')[-1])
        if phase:
            # For phase-based models, look for trained_model_{phase}.h5
            model_file = os.path.join(model_dir, f'trained_model_{phase}.h5')
        else:
            # For non-phase models, look for intersection_agent{num}_model.h5
            model_file = os.path.join(model_dir, f'intersection_agent{agent_num}_model.h5')
            
        if os.path.exists(model_file):
            return model_num, model_file
    
    return None, None

def read_server_config(config_file='server_config_2.ini'):
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
    mapping_config = {
        'location': location_data,
        'map': map_config,
        'visualization': viz_config
    }
    env_file_path = map_config['environment_file'] if map_config['send_topology'] else None
    if env_file_path and not os.path.exists(env_file_path):
        print(f"Warning: Environment file not found at {env_file_path}")
        env_file_path = None
    return server_url, agent_id, mapping_config, env_file_path

class TestingSimulationWithServer(Simulation):
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, 
                 yellow_duration, num_states, num_actions, server_url=None, agent_id=None,
                 mapping_config=None, env_file_path=None):
        # Call the parent constructor
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, 
                         yellow_duration, num_states, num_actions)
        
        # Initialize server communication if URL is provided
        self._server_url = server_url
        self._agent_id = agent_id
        
        if server_url:
            self._communicator = AgentCommunicatorTesting(server_url, agent_id, mapping_config, env_file_path)
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

        # Get initial sync timing if available
        if self._communicator:
            sync_data = self._communicator.get_sync_timing()
            if sync_data:
                self._adjust_timing(sync_data)

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

            # Update server with state and get new sync timing
            if self._communicator:
                # Send current state
                self._communicator.send_state(current_state, self._step, {
                    'queue_length': self._get_queue_length(),
                    'current_phase': traci.trafficlight.getPhase("TL"),
                    'incoming_vehicles': {
                        'N': traci.edge.getLastStepVehicleNumber("N2TL"),
                        'S': traci.edge.getLastStepVehicleNumber("S2TL"),
                        'E': traci.edge.getLastStepVehicleNumber("E2TL"),
                        'W': traci.edge.getLastStepVehicleNumber("W2TL")
                    },
                    'avg_speed': {
                        'N': traci.edge.getLastStepMeanSpeed("N2TL"),
                        'S': traci.edge.getLastStepMeanSpeed("S2TL"),
                        'E': traci.edge.getLastStepMeanSpeed("E2TL"),
                        'W': traci.edge.getLastStepMeanSpeed("W2TL")
                    }
                })

                # Get new sync timing periodically
                if self._step % 60 == 0:  # Check for new sync timing every minute
                    sync_data = self._communicator.get_sync_timing()
                    if sync_data:
                        self._adjust_timing(sync_data)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-config', type=str, default='server_config_1.ini')
    parser.add_argument('--phase', type=str, help='Phase to use for model loading (e.g., "base", "sync"). If not specified, will use non-phase model.')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive testing mode with UI')
    args = parser.parse_args()
    
    # Configure the test
    config = import_test_configuration(config_file='testing_settings.ini')
    
    # Override interactive setting from command line
    config['interactive_testing'] = args.interactive
    
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    
    # Read server configuration first to get agent ID
    server_url, agent_id, mapping_config, env_file_path = read_server_config(args.server_config)
    
    # Find the latest model for this agent
    models_dir = config['models_path_name']
    latest_model_num, model_path = get_latest_model_for_agent(models_dir, agent_id, args.phase)
    
    if latest_model_num is None:
        print(f"Error: No model found for agent {agent_id}")
        if args.phase:
            print(f"Tried to find phase-based model: trained_model_{args.phase}.h5")
        else:
            print(f"Tried to find non-phase model: intersection_agent{agent_id.replace('agent', '')}_model.h5")
        sys.exit(1)
    
    # Update config with the latest model number
    config['model_to_test'] = latest_model_num
    
    # Get plot path
    plot_path = os.path.join(models_dir, f'plots_{latest_model_num}')

    # Print model information
    print("\n=== Model Information ===")
    print(f"Agent ID: {agent_id}")
    print(f"Using model: {latest_model_num}")
    print(f"Model path: {model_path}")
    print(f"Plot path: {plot_path}")
    if args.phase == 'base':
        print("Using base model")
    else:
        print("Using sync-aware model (requires sync_agent)")
    print(f"Testing mode: {'Interactive' if args.interactive else 'Non-interactive'}")
    print("=======================\n")

    if server_url:
        print(f"Connecting to central server at {server_url} as agent {agent_id}")
    else:
        print("Running in standalone mode (no central server)")

    # Create model
    Model = TestModel(
        config['num_states'],
        model_path,
        phase=args.phase
    )

    if args.interactive:
        # Use interactive simulation with UI and random vehicle spawning
        print("[INFO] Running in INTERACTIVE TESTING mode (UI + random vehicle spawning)")
        simulation = InteractiveSimulation(
            Model,
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
        print("----- Testing episode (interactive)")
        simulation_time = simulation.run(config['episode_seed'])
        print("Simulation time:", simulation_time, "s")
        reward_episode = simulation.reward_episode
        print("Average reward:", np.mean(reward_episode))
        print("Total reward:", np.sum(reward_episode))
        queue_length_episode = simulation.queue_length_episode
        print("Average queue length:", np.mean(queue_length_episode))
        print("End of testing")
        simulation.cleanup()
    else:
        # Use the default server testing simulation
        print("Using default server testing simulation")
        TrafficGen = TrafficGenerator(
            config['max_steps'], 
            config['n_cars_generated']
        )
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
            agent_id,
            mapping_config,
            env_file_path
        )
        print("----- Testing episode")
        simulation_time = Simulation.run(config['episode_seed'])
        print("Simulation time:", simulation_time, "s")
        reward_episode = Simulation.reward_episode
        print("Average reward:", np.mean(reward_episode))
        print("Total reward:", np.sum(reward_episode))
        queue_length_episode = Simulation.queue_length_episode
        print("Average queue length:", np.mean(queue_length_episode))
        print("End of testing")
        Simulation.cleanup()