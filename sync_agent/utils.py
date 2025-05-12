import numpy as np
from collections import deque
import random
import json
import os

class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity=100000, batch_size=64):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            batch_size: Number of experiences to sample in each batch
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """Sample a batch of experiences"""
        # Ensure we have enough samples
        if len(self.buffer) < self.batch_size:
            # If not enough samples, return a smaller batch or None
            if len(self.buffer) < 2:
                return None
            batch_size = len(self.buffer)
        else:
            batch_size = self.batch_size
        
        # Sample random experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Organize batch into separate arrays
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences]).reshape(-1, 1)
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences]).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def save_buffer(self, path):
        """Save buffer to disk (optional for long-term learning)"""
        data = []
        for state, action, reward, next_state, done in self.buffer:
            data.append({
                'state': state.tolist(),
                'action': action.tolist(),
                'reward': float(reward),
                'next_state': next_state.tolist(),
                'done': bool(done)
            })
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load_buffer(self, path):
        """Load buffer from disk"""
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Clear current buffer
            self.buffer.clear()
            
            # Load saved experiences
            for exp in data:
                self.buffer.append((
                    np.array(exp['state'], dtype=np.float32),
                    np.array(exp['action'], dtype=np.float32),
                    float(exp['reward']),
                    np.array(exp['next_state'], dtype=np.float32),
                    bool(exp['done'])
                ))
            
            return True
        except Exception as e:
            print(f"Error loading buffer: {e}")
            return False