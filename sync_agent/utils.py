import numpy as np
from collections import deque
import random
import json
import os

def pad_state(state, max_state_dim=256):
    state = np.asarray(state)
    if state.shape[0] < max_state_dim:
        padding = np.zeros((max_state_dim - state.shape[0],))
        state = np.concatenate([state, padding])
    elif state.shape[0] > max_state_dim:
        state = state[:max_state_dim]
    return state

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
        self.action_dim = None  # Will be set when first action is added
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        # Convert action to numpy array if it isn't already
        action = np.asarray(action, dtype=np.float32)
        
        # Set action_dim if not set
        if self.action_dim is None:
            self.action_dim = action.shape[0] if action.ndim > 0 else 1
            print(f"Setting action_dim to {self.action_dim}")
        
        # Ensure action has correct shape
        if action.ndim == 0:
            action = np.array([action], dtype=np.float32)
        elif action.shape[0] != self.action_dim:
            print(f"Warning: Action shape mismatch. Expected {self.action_dim}, got {action.shape}. Reshaping...")
            if action.shape[0] > self.action_dim:
                action = action[:self.action_dim]
            else:
                padding = np.zeros(self.action_dim - action.shape[0], dtype=np.float32)
                action = np.concatenate([action, padding])
        
        # Convert done to boolean
        done = bool(done)
        
        self.buffer.append((state, action, float(reward), next_state, done))
    
    def sample(self):
        """Sample a batch of experiences"""
        if len(self.buffer) < self.batch_size:
            return None
            
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        # Debug: Print shapes of first few actions
        print("Action shapes in batch:")
        for i, exp in enumerate(experiences[:3]):  # Print first 3 actions
            print(f"Action {i} shape: {exp[1].shape}")
        
        # Infer max_state_dim from the first state in the batch, or use 256 as default
        first_state = experiences[0][0]
        max_state_dim = 256
        if hasattr(first_state, 'shape') and len(first_state.shape) > 0:
            max_state_dim = first_state.shape[0] if first_state.shape[0] > 0 else 256
            
        states = np.array([pad_state(exp[0], max_state_dim) for exp in experiences], dtype=np.float32)
        
        # Ensure all actions have the same shape before creating the array
        actions = []
        for exp in experiences:
            action = exp[1]
            if action.shape[0] != self.action_dim:
                if action.shape[0] > self.action_dim:
                    action = action[:self.action_dim]
                else:
                    padding = np.zeros(self.action_dim - action.shape[0], dtype=np.float32)
                    action = np.concatenate([action, padding])
            actions.append(action)
        actions = np.array(actions, dtype=np.float32)
        
        rewards = np.array([exp[2] for exp in experiences], dtype=np.float32)
        next_states = np.array([pad_state(exp[3], max_state_dim) for exp in experiences], dtype=np.float32)
        dones = np.array([exp[4] for exp in experiences], dtype=np.bool_)
        
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