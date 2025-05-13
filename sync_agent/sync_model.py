import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Reshape, Flatten
from tensorflow.keras.models import Model
import numpy as np

class SyncDRLModel:
    """Deep Reinforcement Learning model for traffic signal synchronization"""
    
    def __init__(
        self, 
        state_dim, 
        action_dim,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        hidden_sizes=(256, 256),
        max_intersections=10  # Maximum number of intersections to support
    ):
        """
        Initialize the Soft Actor-Critic model
        
        Args:
            state_dim: dimension of state space
            action_dim: dimension of action space
            actor_learning_rate: learning rate for actor network
            critic_learning_rate: learning rate for critic network
            gamma: discount factor
            tau: target network update rate
            hidden_sizes: tuple of hidden layer sizes
            max_intersections: maximum number of intersections to support
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.max_intersections = max_intersections
        
        # Calculate dimensions for fixed-size representation
        self.features_per_intersection = 4  # traffic volume, queue length, waiting time, cycle time
        self.features_per_pair = 3  # distance, travel time, current offset
        
        # Calculate maximum possible state dimension
        self.max_state_dim = (self.max_intersections * self.features_per_intersection + 
                            (self.max_intersections * (self.max_intersections - 1)) // 2 * self.features_per_pair)
        
        # Initialize actor and critic networks
        self.actor = self._build_actor(hidden_sizes)
        self.critic_1 = self._build_critic(hidden_sizes)
        self.critic_2 = self._build_critic(hidden_sizes)
        
        # Initialize target networks
        self.target_critic_1 = self._build_critic(hidden_sizes)
        self.target_critic_2 = self._build_critic(hidden_sizes)
        
        # Copy weights
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # Set up optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        
        # Set up training metrics
        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        self.critic_1_loss_metric = tf.keras.metrics.Mean('critic_1_loss', dtype=tf.float32)
        self.critic_2_loss_metric = tf.keras.metrics.Mean('critic_2_loss', dtype=tf.float32)
    
    def _preprocess_state(self, state):
        """Preprocess state to handle variable dimensions"""
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        
        # Pad or truncate state to max_state_dim
        if state.shape[1] < self.max_state_dim:
            padding = np.zeros((state.shape[0], self.max_state_dim - state.shape[1]))
            state = np.concatenate([state, padding], axis=1)
        elif state.shape[1] > self.max_state_dim:
            state = state[:, :self.max_state_dim]
        
        return state
    
    def _build_actor(self, hidden_sizes):
        """Build the actor network (policy)"""
        inputs = Input(shape=(self.max_state_dim,))
        x = inputs
        
        # Calculate features per intersection
        features_per_intersection = self.features_per_intersection + (self.max_intersections - 1) * self.features_per_pair
        
        # Reshape to handle variable number of intersections
        x = Reshape((self.max_intersections, features_per_intersection))(x)
        x = Flatten()(x)
        
        for size in hidden_sizes:
            x = Dense(size)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        # Output layer (normalized to 0-1)
        outputs = Dense(self.action_dim, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='actor')
    
    def _build_critic(self, hidden_sizes):
        """Build critic network (Q-value function)"""
        state_inputs = Input(shape=(self.max_state_dim,))
        action_inputs = Input(shape=(self.action_dim,))
        
        # Calculate features per intersection
        features_per_intersection = self.features_per_intersection + (self.max_intersections - 1) * self.features_per_pair
        
        # Reshape state inputs
        x = Reshape((self.max_intersections, features_per_intersection))(state_inputs)
        x = Flatten()(x)
        
        # Combine with action inputs
        x = tf.concat([x, action_inputs], axis=1)
        
        for size in hidden_sizes:
            x = Dense(size)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        # Output Q-value
        outputs = Dense(1)(x)
        
        return Model([state_inputs, action_inputs], outputs, name='critic')
    
    def policy(self, state, deterministic=False, noise_scale=0.1):
        """
        Get action from policy/actor network
        
        Args:
            state: Current state
            deterministic: If True, return deterministic action, else add exploration noise
            noise_scale: Scale of exploration noise
            
        Returns:
            Action vector
        """
        # Preprocess state to handle variable dimensions
        state = self._preprocess_state(state)
        
        # Get action from policy
        action = self.actor.predict(state)[0]
        
        # Add exploration noise if not deterministic
        if not deterministic:
            noise = np.random.normal(0, noise_scale, size=self.action_dim)
            action = np.clip(action + noise, 0.0, 1.0)
        
        return action
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """Single training step for actor and critic networks"""
        # Preprocess states
        states = self._preprocess_state(states)
        next_states = self._preprocess_state(next_states)
        
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape_1, tf.GradientTape() as critic_tape_2:
            # Get next actions and log probs from current policy
            next_actions = self.actor(next_states)
            
            # Compute target Q values
            target_q1 = self.target_critic_1([next_states, next_actions])
            target_q2 = self.target_critic_2([next_states, next_actions])
            
            # Take minimum of both critics to mitigate overestimation
            target_q = tf.minimum(target_q1, target_q2)
            
            # Compute target value (Bellman equation)
            q_target = rewards + (1 - dones) * self.gamma * target_q
            
            # Get current Q estimates
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            
            # Compute critic losses (MSE)
            critic_loss_1 = tf.reduce_mean(tf.square(current_q1 - q_target))
            critic_loss_2 = tf.reduce_mean(tf.square(current_q2 - q_target))
            
            # Compute actor loss
            actor_actions = self.actor(states)
            q1 = self.critic_1([states, actor_actions])
            actor_loss = -tf.reduce_mean(q1)  # Negative because we want to maximize Q
        
        # Compute critic gradients
        critic_grad_1 = critic_tape_1.gradient(critic_loss_1, self.critic_1.trainable_variables)
        critic_grad_2 = critic_tape_2.gradient(critic_loss_2, self.critic_2.trainable_variables)
        
        # Compute actor gradients
        actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        
        # Apply gradients
        self.critic_optimizer.apply_gradients(zip(critic_grad_1, self.critic_1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grad_2, self.critic_2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        
        # Update metrics
        self.actor_loss_metric.update_state(actor_loss)
        self.critic_1_loss_metric.update_state(critic_loss_1)
        self.critic_2_loss_metric.update_state(critic_loss_2)
        
        return actor_loss, critic_loss_1, critic_loss_2
    
    def train(self, batch):
        """
        Train the model on a batch of experiences
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            
        Returns:
            Dictionary of loss metrics
        """
        states, actions, rewards, next_states, dones = batch
        
        # Reset metrics
        self.actor_loss_metric.reset_states()
        self.critic_1_loss_metric.reset_states()
        self.critic_2_loss_metric.reset_states()
        
        # Perform training step
        actor_loss, critic_loss_1, critic_loss_2 = self._train_step(
            states, actions, rewards, next_states, dones
        )
        
        # Update target networks with soft update
        self._update_target_networks()
        
        return {
            'actor_loss': self.actor_loss_metric.result().numpy(),
            'critic_1_loss': self.critic_1_loss_metric.result().numpy(),
            'critic_2_loss': self.critic_2_loss_metric.result().numpy()
        }
    
    def _update_target_networks(self):
        """Soft update target networks"""
        for target, source in zip(self.target_critic_1.variables, self.critic_1.variables):
            target.assign((1 - self.tau) * target + self.tau * source)
        for target, source in zip(self.target_critic_2.variables, self.critic_2.variables):
            target.assign((1 - self.tau) * target + self.tau * source)
    
    def save_models(self, path):
        """Save model weights"""
        self.actor.save_weights(f"{path}_actor.h5")
        self.critic_1.save_weights(f"{path}_critic1.h5")
        self.critic_2.save_weights(f"{path}_critic2.h5")
    
    def load_models(self, path):
        """Load model weights"""
        try:
            self.actor.load_weights(f"{path}_actor.h5")
            self.critic_1.load_weights(f"{path}_critic1.h5")
            self.critic_2.load_weights(f"{path}_critic2.h5")
            self.target_critic_1.set_weights(self.critic_1.get_weights())
            self.target_critic_2.set_weights(self.critic_2.get_weights())
            return True
        except:
            return False