from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot
import os
import sys
import datetime
import numpy as np
import argparse

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

def train_base_model(config, continue_from=None):
    """
    Train the base model without server/sync
    
    Args:
        config: training configuration
        continue_from: path to previous model to continue training from (optional)
    """
    print("\n" + "="*50)
    print("STARTING BASE TRAINING")
    print("="*50)
    
    if continue_from:
        print(f"\nAttempting to continue training from: {continue_from}")
    else:
        print("\nStarting training from scratch (no previous model)")
    
    # Set up SUMO
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    # Create model
    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        config['num_states'], 
        config['num_actions']
    )
    
    # Try to load previous model if specified
    if continue_from:
        if os.path.exists(continue_from):
            print(f"Found previous model at: {continue_from}")
            if Model.load_base_model(continue_from):
                print("✓ Successfully loaded previous model")
            else:
                print("✗ Failed to load previous model, starting from scratch")
        else:
            print(f"✗ Previous model not found at: {continue_from}, starting from scratch")
    
    # Create memory and traffic generator
    memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    # Create simulation (without server connection)
    simulation = Simulation(
        Model,
        memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    # Training loop
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Base Training: Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation_time, training_time = simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', 
              round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n" + "="*50)
    print("BASE TRAINING FINISHED")
    print("="*50)
    print("Starting time:", timestamp_start)
    print("Ending time:", datetime.datetime.now())
    print("Session info saved at:", path)

    # Save base model
    model_path = os.path.join(path, 'trained_model_base.h5')
    print(f"\nSaving base model to: {model_path}")
    Model.save_model(path, phase='base')
    
    # Cleanup
    simulation.cleanup()
    
    return path

def main():
    parser = argparse.ArgumentParser(description='Train base traffic light control model')
    parser.add_argument('--continue-from', type=str, default=None,
                       help='Path to previous model to continue training from (optional)')
    args = parser.parse_args()
    
    # Load training configuration
    config = import_train_configuration(config_file='training_settings.ini')
    
    # Run base training
    train_base_model(config, continue_from=args.continue_from)

if __name__ == "__main__":
    main()