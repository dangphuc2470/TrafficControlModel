import os
import sys
import time
import logging
import argparse
import json
import numpy as np
from sync_model import SyncDRLModel
from sync_environment import IntersectionSyncEnv
from utils import ReplayBuffer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sync_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SyncTest")

class SyncController:
    """Controller for synchronizing multiple intersections using a single trained model"""
    
    def __init__(self, model_path, num_intersections=4):
        # State dimension: 4 local features + (3 sync features * 3 neighbors) = 13
        self.state_dim = 13
        # Action dimension: 1 (single offset value)
        self.action_dim = 1
        
        # Initialize model
        self.model = SyncDRLModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            hidden_sizes=(256, 256)
        )
        
        # Load trained model
        self.model.load_models(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Initialize environment
        self.env = IntersectionSyncEnv()
        self.num_intersections = num_intersections
        
        # Store intersection data
        self.intersection_data = {}
        for i in range(num_intersections):
            self.intersection_data[f"agent{i+1}"] = {
                "states": [],
                "queue_lengths": [],
                "waiting_times": []
            }
    
    def update_intersection_data(self, new_data):
        """Update data for all intersections"""
        self.intersection_data = new_data
        self.env.update_intersection_data(new_data)
    
    def get_actions(self):
        """Get actions for all intersections"""
        actions = {}
        
        # Get state for all intersections
        state = self.env._get_state()
        
        # Get action from model (deterministic for testing)
        action = self.model.policy(state, deterministic=True)
        
        # Get cycle times from environment
        cycle_times = self.env.cycle_times
        
        # Distribute and scale action to all intersections
        for i, intersection_id in enumerate(self.intersection_data.keys()):
            # Get cycle time for this intersection (default to 60 seconds if not available)
            cycle_time = cycle_times.get(intersection_id, 60)
            
            # Scale action to cycle time (0-1 -> 0-cycle_time)
            scaled_action = action[0] * cycle_time
            
            # Store scaled action
            actions[intersection_id] = scaled_action
            
            logger.info(f"Generated offset for {intersection_id}: {scaled_action:.2f}s (cycle time: {cycle_time}s)")
        
        return actions
    
    def apply_actions(self, actions):
        """Apply actions to environment and get new states"""
        # Convert actions dictionary to a single action array
        action_array = np.array([action for action in actions.values()])
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = self.env.step(action_array)
        
        # Store new state and metrics
        new_states = {id: next_state for id in self.intersection_data.keys()}
        metrics = {id: info['metrics'] for id in self.intersection_data.keys()}
        
        return new_states, metrics

def main():
    # Path to trained model
    model_path = "sync_models/model_final"
    sync_times_path = "../central_server/server_data/sync_times.json"
    
    # Initialize controller
    controller = SyncController(model_path, num_intersections=4)
    
    # Load initial test data
    with open("../central_server/server_data/agent_data.json", 'r') as f:
        test_data = json.load(f)
    
    # Update controller with initial test data
    controller.update_intersection_data(test_data)
    
    # Run continuously until all agents disconnect
    logger.info("Starting continuous synchronization monitoring...")
    update_interval = 10  # seconds between updates
    
    try:
        while True:
            # Check if any agents are still connected
            if not test_data:
                logger.info("No agents connected, stopping synchronization")
                break
                
            # Get actions for all intersections
            actions = controller.get_actions()
            logger.info(f"Generated synchronization parameters: {actions}")
            
            # Format sync times for saving
            sync_times = {}
            for agent1 in test_data.keys():
                sync_times[agent1] = {}
                for agent2 in test_data.keys():
                    if agent1 != agent2:
                        # Get distance and travel time from environment
                        pair = tuple(sorted([agent1, agent2]))
                        # Try both tuple orders for distance and travel time
                        distance = (controller.env.distances.get(pair, 0.0) or 
                                  controller.env.distances.get((agent1, agent2), 0.0) or
                                  controller.env.distances.get((agent2, agent1), 0.0))
                        travel_time = (controller.env.travel_times.get(pair, 0.0) or
                                     controller.env.travel_times.get((agent1, agent2), 0.0) or
                                     controller.env.travel_times.get((agent2, agent1), 0.0))
                        cycle_time = controller.env.cycle_times.get(agent1, 60)
                        offset = actions[agent1]
                        
                        # Log the values for debugging
                        logger.info(f"Pair {agent1}-{agent2}:")
                        logger.info(f"  Distance: {distance:.2f} km")
                        logger.info(f"  Travel time: {travel_time:.2f} s")
                        logger.info(f"  Cycle time: {cycle_time:.2f} s")
                        logger.info(f"  Offset: {offset:.2f} s")
                        
                        # Calculate average speed from agent data
                        avg_speed = 40.0  # Default
                        if agent1 in test_data and 'states' in test_data[agent1] and test_data[agent1]['states']:
                            latest_state = test_data[agent1]['states'][-1]
                            if 'traffic_data' in latest_state and 'avg_speed' in latest_state['traffic_data']:
                                speeds = latest_state['traffic_data']['avg_speed'].values()
                                if speeds:
                                    avg_speed = sum(speeds) / len(speeds)
                        
                        sync_times[agent1][agent2] = {
                            "distance_km": round(distance, 2),
                            "travel_time_sec": round(travel_time, 2),
                            "optimal_offset_sec": round(offset, 2),
                            "cycle_time_sec": cycle_time,
                            "drl_optimized": True,
                            "avg_speed_kmh": round(avg_speed, 2)
                        }
            
            # Save sync times to file
            try:
                with open(sync_times_path, 'w') as f:
                    json.dump(sync_times, f, indent=2)
                logger.info(f"Saved sync times to {sync_times_path}")
            except Exception as e:
                logger.error(f"Error saving sync times: {e}")
            
            # Apply actions and get new states
            new_states, metrics = controller.apply_actions(actions)
            
            # Log results
            for intersection_id, metric in metrics.items():
                logger.info(f"Intersection {intersection_id}:")
                logger.info(f"  - Average waiting time: {metric['avg_waiting_time']:.2f}")
                logger.info(f"  - Average queue length: {metric['avg_queue_length']:.2f}")
            
            # Wait for next update
            time.sleep(update_interval)
            
            # Reload test data to check for agent disconnections
            try:
                with open("../central_server/server_data/agent_data.json", 'r') as f:
                    test_data = json.load(f)
                controller.update_intersection_data(test_data)
            except Exception as e:
                logger.error(f"Error reading agent data: {e}")
                break
                
    except KeyboardInterrupt:
        logger.info("Synchronization monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in synchronization monitoring: {e}")
    finally:
        logger.info("Synchronization monitoring stopped")

if __name__ == "__main__":
    main() 