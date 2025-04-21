import requests
import json
import time
import socket
import threading
import os
import numpy as np

class AgentCommunicator:
    def __init__(self, server_url, agent_id=None):
        """
        Initialize the communicator with the server URL
        
        Args:
            server_url: URL of the central server
            agent_id: Unique ID for this agent (if None, hostname will be used)
        """
        self.server_url = server_url
        self.agent_id = agent_id or socket.gethostname()
        self.data = {
            'agent_id': self.agent_id,
            'rewards': [],
            'queue_lengths': [],
            'waiting_times': [],
            'status': 'initializing',
            'last_episode': -1,
            'config': {},
            'model_info': {}
        }
        self.last_sync = 0
        self.sync_interval = 30  # seconds
        self.background_thread = None
        self.running = False
        
        # Create a directory to store data locally in case of connection issues
        self.backup_dir = f'agent_{self.agent_id}_data'
        os.makedirs(self.backup_dir, exist_ok=True)
        
        print(f"Agent communicator initialized with ID: {self.agent_id}")
    
    def start_background_sync(self):
        """Start a background thread to periodically sync with the server"""
        if self.background_thread is not None and self.background_thread.is_alive():
            return  # Already running
        
        self.running = True
        self.background_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.background_thread.start()
        
    def stop_background_sync(self):
        """Stop the background sync thread"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
    
    def _sync_loop(self):
        """Background thread function to periodically sync with server"""
        while self.running:
            try:
                self.sync_with_server()
            except Exception as e:
                print(f"Error in background sync: {e}")
                # Backup data locally
                self._backup_data()
            
            # Sleep until next sync
            time.sleep(self.sync_interval)
    
    def _backup_data(self):
        """Save data locally as backup in case of server connection issues"""
        backup_file = os.path.join(self.backup_dir, f'backup_{int(time.time())}.json')
        with open(backup_file, 'w') as f:
            json.dump(self.data, f)
    
    def update_episode_result(self, episode, reward, queue_length, waiting_time=None):
        """
        Update results for a specific episode
        
        Args:
            episode: Episode number
            reward: Total reward for the episode
            queue_length: Average queue length
            waiting_time: Total waiting time (optional)
        """
        self.data['last_episode'] = episode
        self.data['rewards'].append(float(reward))
        self.data['queue_lengths'].append(float(queue_length))
        
        if waiting_time is not None:
            self.data['waiting_times'].append(float(waiting_time))
        
        # If it's been long enough since last sync, sync now
        current_time = time.time()
        if current_time - self.last_sync >= self.sync_interval:
            self.sync_with_server()
    
    def update_status(self, status):
        """Update the agent's status"""
        self.data['status'] = status
        
    def update_config(self, config):
        """Update the agent's configuration"""
        self.data['config'] = config
        
    def update_model_info(self, model_info):
        """Update information about the model"""
        self.data['model_info'] = model_info
    
    def sync_with_server(self):
        """Send accumulated data to the central server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/update", 
                json=self.data,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Successfully synced data with server. Episodes: {len(self.data['rewards'])}")
                self.last_sync = time.time()
                return True
            else:
                print(f"Server sync failed with status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error during sync: {e}")
            return False