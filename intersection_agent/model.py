import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self._is_sync_training = False  # Flag to track training phase
        self._rewards = []
        self._delays = []
        self._queues = []


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        # Use the legacy optimizer to avoid the M1/M2 Mac performance issues
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self._learning_rate)
        model.compile(loss=losses.mean_squared_error, optimizer=self.optimizer)
        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path, phase='base', model_name=None):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        
        Args:
            path: Directory path to save the model
            phase: Training phase ('base' or 'sync')
            model_name: Optional custom name for the model file (if not provided, will use trained_model_{phase}.h5)
        """
        if model_name is None:
            model_name = f'trained_model_{phase}.h5'
            
        # Save the model
        model_path = os.path.join(path, model_name)
        self._model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Try to save the model plot
        try:
            plot_path = os.path.join(path, f'model_structure_{phase}.png')
            plot_model(self._model, to_file=plot_path, 
                      show_shapes=True, show_layer_names=True)
            print(f"Model structure plot saved to: {plot_path}")
        except Exception as e:
            print(f"Could not generate model plot: {str(e)}")
            print("To enable model plotting, install graphviz:")
            print("  - On macOS: brew install graphviz")
            print("  - On Ubuntu: sudo apt-get install graphviz")
            print("  - On Windows: Download from https://graphviz.org/download/")
            
        # Generate training plots
        try:
            # Create plots directory if it doesn't exist
            plots_dir = os.path.join(path, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot rewards
            if hasattr(self, '_rewards'):
                plt.figure(figsize=(10, 6))
                plt.plot(self._rewards)
                plt.title('Training Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.savefig(os.path.join(plots_dir, 'rewards.png'))
                plt.close()
                print(f"Rewards plot saved to: {os.path.join(plots_dir, 'rewards.png')}")
            
            # Plot delays
            if hasattr(self, '_delays'):
                plt.figure(figsize=(10, 6))
                plt.plot(self._delays)
                plt.title('Average Delay')
                plt.xlabel('Episode')
                plt.ylabel('Delay (s)')
                plt.savefig(os.path.join(plots_dir, 'delays.png'))
                plt.close()
                print(f"Delays plot saved to: {os.path.join(plots_dir, 'delays.png')}")
            
            # Plot queue lengths
            if hasattr(self, '_queues'):
                plt.figure(figsize=(10, 6))
                plt.plot(self._queues)
                plt.title('Average Queue Length')
                plt.xlabel('Episode')
                plt.ylabel('Queue Length')
                plt.savefig(os.path.join(plots_dir, 'queues.png'))
                plt.close()
                print(f"Queue lengths plot saved to: {os.path.join(plots_dir, 'queues.png')}")
                
        except Exception as e:
            print(f"Could not generate training plots: {str(e)}")


    def load_base_model(self, path):
        """
        Load the base model for sync-aware training
        """
        if os.path.isfile(path):
            try:
                self._model = load_model(path)
                self._is_sync_training = True
                # Reduce learning rate for fine-tuning
                self._learning_rate *= 0.1
                self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self._learning_rate)
                self._model.compile(loss=losses.mean_squared_error, optimizer=self.optimizer)
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        return False


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


    @property
    def is_sync_training(self):
        return self._is_sync_training


    def update_metrics(self, reward, delay, queue):
        """
        Update training metrics for plotting
        
        Args:
            reward: Current episode reward
            delay: Current episode average delay
            queue: Current episode average queue length
        """
        if not hasattr(self, '_rewards'):
            self._rewards = []
        if not hasattr(self, '_delays'):
            self._delays = []
        if not hasattr(self, '_queues'):
            self._queues = []
            
        self._rewards.append(reward)
        self._delays.append(delay)
        self._queues.append(queue)


class TestModel:
    def __init__(self, input_dim, model_path, phase=None):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path, phase)

    def _load_my_model(self, model_folder_path, phase):
        """
        Load the model stored in the folder specified by the model number
        Args:
            model_folder_path: Path to the model folder
            phase: If None, load non-phase model. If specified, load phase-based model
        """
        if phase is None:
            # Try to load non-phase model first
            model_file_path = model_folder_path
            if os.path.isfile(model_file_path):
                return load_model(model_file_path)
        else:
            # Try to load phase-based model
            model_file_path = os.path.join(model_folder_path, f'trained_model_{phase}.h5')
            if os.path.isfile(model_file_path):
                return load_model(model_file_path)
        
        # If we get here, try the other approach
        if phase is None:
            # If non-phase failed, try phase-based
            model_file_path = os.path.join(model_folder_path, 'trained_model_base.h5')
            if os.path.isfile(model_file_path):
                print("Warning: Using phase-based model as fallback")
                return load_model(model_file_path)
        else:
            # If phase-based failed, try non-phase
            if os.path.isfile(model_folder_path):
                print("Warning: Using non-phase model as fallback")
                return load_model(model_folder_path)
        
        # If all attempts fail
        sys.exit(f"Model not found at {model_folder_path}")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    @property
    def input_dim(self):
        return self._input_dim