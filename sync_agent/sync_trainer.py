import os
import sys
import time
import threading
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from sync_environment import IntersectionSyncEnv
from sync_model import SyncDRLModel
from utils import ReplayBuffer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sync_trainer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SyncTrainer")

class SyncTrainer:
    """
    Trainer for the synchronization DRL model.
    
    Integrates with the central server to:
    1. Collect traffic data from intersection agents
    2. Train the DRL model
    3. Provide synchronization parameters back to agents
    """
    
    def __init__(
        self,
        model_dir="sync_models",
        data_path="../central_server_old/server_data/agent_data.json",
        output_path="../central_server_old/server_data/sync_times.json",
        update_interval=60,  # seconds between updates
        train_interval=30,   # seconds between training batches
        save_interval=3600,  # seconds between model saves
        batch_size=64,
        buffer_capacity=100000,
        min_buffer_size=1000,
        sync_with_agents=True
    ):
        """
        Initialize the synchronization trainer
        
        Args:
            model_dir: Directory to save/load models
            data_path: Path to agent data JSON file from central server
            output_path: Path to save synchronized timing data for agents
            update_interval: How often to update from agent data (seconds)
            train_interval: How often to train the model (seconds)
            save_interval: How often to save the model (seconds)
            batch_size: Training batch size
            buffer_capacity: Replay buffer capacity
            min_buffer_size: Minimum buffer size before starting training
            sync_with_agents: Whether to sync timing with actual agents
        """
        # Convert model_dir to absolute path
        self.model_dir = os.path.abspath(model_dir)
        self.data_path = data_path
        self.output_path = output_path
        self.update_interval = update_interval
        self.train_interval = train_interval
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.sync_with_agents = sync_with_agents
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize environment and agent data
        self.agent_data = {}
        self.env = IntersectionSyncEnv()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)
        
        # Initialize model (defer until we know the state/action dimensions)
        self.model = None
        
        # Initialize synchronization data
        self.sync_times = {}
        
        # Training stats
        self.episode_rewards = []
        self.episode_metrics = []
        self.avg_waiting_times = []
        self.avg_queue_lengths = []
        
        # Thread control
        self.is_running = False
        self.threads = []
        
        # Flags for tracking changes in agent topology
        self.topology_changed = False
        self.last_sync_save = 0
        self.last_model_save = time.time()  # Initialize last_model_save
        
        logger.info(f"SyncTrainer initialized with model directory: {self.model_dir}")
    
    def start(self):
        """Start the training process in background threads"""
        if self.is_running:
            logger.warning("Trainer is already running")
            return
        
        self.is_running = True
        
        # Start update thread
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()
        self.threads.append(update_thread)
        
        # Start training thread
        train_thread = threading.Thread(target=self._train_loop, daemon=True)
        train_thread.start()
        self.threads.append(train_thread)
        
        logger.info("Training started in background threads")
    
    def stop(self):
        """Stop all running threads"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.threads = []
        
        # Save model and buffer
        if self.model is not None:
            self._save_model()
            self._save_buffer()
        
        logger.info("Training stopped")
    
    def _update_loop(self):
        """Background thread for updating from agent data"""
        while self.is_running:
            try:
                # Update agent data from central server
                new_topology = self._update_agent_data()
                
                # If topology changed, we need to reinitialize environment and model
                if new_topology:
                    self._reinitialize_model()
                
                # Update environment with latest data
                self.env.update_intersection_data(self.agent_data)
                
                # Run the environment for a step to generate new synchronization data
                if self.model is not None:
                    self._generate_sync_data()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}", exc_info=True)
                time.sleep(10)  # Sleep on error to prevent rapid retries
    
    def _train_loop(self):
        """Background thread for training the model"""
        while self.is_running:
            try:
                # Skip if model not initialized
                if self.model is None:
                    time.sleep(5)
                    continue
                
                # Train if buffer has enough samples
                if len(self.replay_buffer) >= self.min_buffer_size:
                    # Sample batch and train
                    batch = self.replay_buffer.sample()
                    if batch is not None:
                        losses = self.model.train(batch)
                        logger.info(f"Training step - Actor Loss: {losses['actor_loss']:.4f}, "
                                    f"Critic Loss: {losses['critic_1_loss']:.4f}")
                
                # Save mzzodel periodically
                current_time = time.time()
                if current_time - self.last_model_save > self.save_interval:
                    self._save_model()
                    self._save_buffer()
                    self.last_model_save = current_time
                
                # Sleep until next train step
                time.sleep(self.train_interval)
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}", exc_info=True)
                time.sleep(10)  # Sleep on error to prevent rapid retries




    def _update_agent_data(self):
        """
        Update agent data from central server
        
        Returns:
            bool: True if topology changed, False otherwise
        """
        try:
            # Check if data file exists
            if not os.path.exists(self.data_path):
                logger.warning(f"Agent data file not found: {self.data_path}")
                return False
            
            # Load agent data
            with open(self.data_path, 'r') as f:
                new_agent_data = json.load(f)
            
            # Check for topology changes
            topology_changed = False
            
            # Case 1: New agents
            if set(new_agent_data.keys()) != set(self.agent_data.keys()):
                topology_changed = True
            
            # Case 2: Topology data changed for existing agents
            for agent_id, data in new_agent_data.items():
                if agent_id in self.agent_data:
                    if 'topology' in data and ('topology' not in self.agent_data[agent_id] or 
                                             data['topology'] != self.agent_data[agent_id]['topology']):
                        topology_changed = True
                        break
            
            # Update agent data
            self.agent_data = new_agent_data
            
            # Log status
            logger.info(f"Updated agent data - {len(self.agent_data)} agents")
            if topology_changed:
                logger.info("Topology changed, reinitializing model")
            
            return topology_changed
            
        except Exception as e:
            logger.error(f"Error updating agent data: {e}")
            return False
    
    def _reinitialize_model(self):
        """Reinitialize model when topology changes"""
        # Reset environment with new agent data
        self.env = IntersectionSyncEnv(self.agent_data)
        
        # Get state and action dimensions from environment
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Initialize model with new dimensions
        self.model = SyncDRLModel(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            hidden_sizes=(256, 256)
        )
        
        # Try to load existing model
        model_path = os.path.join(self.model_dir, "sync_model")
        if not self.model.load_models(model_path):
            logger.info("No existing model found, using new model")
        
        # Reset buffer
        buffer_path = os.path.join(self.model_dir, "replay_buffer.json")
        if not self.replay_buffer.load_buffer(buffer_path):
            logger.info("No existing buffer found, using new buffer")
        
        logger.info(f"Model reinitialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    def _generate_sync_data(self):
        """
        Generate synchronization data using the current model
        
        This runs the environment for one step to get new sync times
        """
        # Skip if no agent data
        if not self.agent_data:
            return
        
        # Reset environment and get state
        state, _ = self.env.reset()
        
        # Get action from policy (no exploration for production)
        action = self.model.policy(state, deterministic=True)
        
        # Take a step in the environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Store experience in replay buffer (for training)
        self.replay_buffer.add(state, action, reward, next_state, False)
        
        # Store metrics
        self.episode_rewards.append(reward)
        self.episode_metrics.append(info['metrics'])
        
        # Get the new optimal offsets
        self.sync_times = self._format_sync_times(info['offsets'])
        
        # Save sync times to output file
        self._save_sync_times()
        
        # Log occasional progress
        if len(self.episode_rewards) % 10 == 0:
            logger.info(f"Episode {len(self.episode_rewards)} - "
                       f"Reward: {reward:.2f}, "
                       f"Avg Waiting Time: {info['metrics']['avg_waiting_time']:.2f}")
    
    def _format_sync_times(self, offsets):
        """
        Format offsets from the environment to the format expected by the central server
        
        Args:
            offsets: Dictionary of offsets {(id1, id2): offset_seconds}
            
        Returns:
            Dictionary in the format expected by central server
        """
        sync_times = {}
        
        # For each intersection
        for id1 in self.agent_data.keys():
            sync_times[id1] = {}
            
            # For each target intersection
            for id2 in self.agent_data.keys():
                if id1 != id2:
                    # Check if we have offset data for this pair
                    pair = tuple(sorted([id1, id2]))
                    if pair in offsets:
                        offset_sec = offsets[pair]
                        
                        # Get basic spatial data
                        distance_km = self.env.distances.get(pair, 0)
                        travel_time_sec = self.env.travel_times.get(pair, 0)
                        
                        # Store in the expected format
                        sync_times[id1][id2] = {
                            "distance_km": round(distance_km, 2),
                            "travel_time_sec": round(travel_time_sec, 2),
                            "optimal_offset_sec": round(offset_sec, 2),
                            "cycle_time_sec": self.env.cycle_times.get(id1, 38),
                            "drl_optimized": True
                        }
        
        return sync_times
    
    def _save_sync_times(self):
        """Save sync times to output file"""
        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.sync_times, f, indent=2)
            logger.debug(f"Saved sync times to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving sync times: {e}")
    
    def _save_model(self):
        """Save model to disk"""
        if self.model is not None:
            try:
                model_path = os.path.join(self.model_dir, "sync_model")
                self.model.save_models(model_path)
                logger.info(f"Saved model to {model_path}")
                
                # Also save training metrics
                self._save_metrics()
                
                # Update last save time
                self.last_model_save = time.time()
            except Exception as e:
                logger.error(f"Error saving model: {e}")
    
    def _save_buffer(self):
        """Save replay buffer to disk"""
        try:
            buffer_path = os.path.join(self.model_dir, "replay_buffer.json")
            self.replay_buffer.save_buffer(buffer_path)
            logger.info(f"Saved replay buffer to {buffer_path}")
        except Exception as e:
            logger.error(f"Error saving replay buffer: {e}")
    
    def _save_metrics(self):
        """Save training metrics and plots"""
        try:
            # Save metrics to CSV
            metrics_path = os.path.join(self.model_dir, "training_metrics.csv")
            with open(metrics_path, 'w') as f:
                f.write("episode,reward,avg_waiting_time,avg_queue_length\n")
                for i, (reward, metrics) in enumerate(zip(self.episode_rewards, self.episode_metrics)):
                    f.write(f"{i},{reward},{metrics['avg_waiting_time']},{metrics['avg_queue_length']}\n")
            
            # Generate plots
            self._generate_plots()
            
            logger.info(f"Saved training metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _generate_plots(self):
        """Generate training plots"""
        try:
            # Extract metrics
            rewards = self.episode_rewards
            waiting_times = [m['avg_waiting_time'] for m in self.episode_metrics]
            queue_lengths = [m['avg_queue_length'] for m in self.episode_metrics]
            
            # Create plots directory
            plots_dir = os.path.join(self.model_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot rewards
            plt.figure(figsize=(10, 6))
            plt.plot(rewards)
            plt.title('Rewards During Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(plots_dir, "rewards.png"))
            plt.close()
            
            # Plot waiting times
            plt.figure(figsize=(10, 6))
            plt.plot(waiting_times)
            plt.title('Average Waiting Time During Training')
            plt.xlabel('Episode')
            plt.ylabel('Average Waiting Time (s)')
            plt.savefig(os.path.join(plots_dir, "waiting_times.png"))
            plt.close()
            
            # Plot queue lengths
            plt.figure(figsize=(10, 6))
            plt.plot(queue_lengths)
            plt.title('Average Queue Length During Training')
            plt.xlabel('Episode')
            plt.ylabel('Average Queue Length')
            plt.savefig(os.path.join(plots_dir, "queue_lengths.png"))
            plt.close()
            
            logger.info(f"Generated training plots in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")