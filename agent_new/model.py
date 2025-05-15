import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

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


    def save_model(self, path, phase='base'):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        model_name = f'trained_model_{phase}.h5'
        self._model.save(os.path.join(path, model_name))
        plot_model(self._model, to_file=os.path.join(path, f'model_structure_{phase}.png'), 
                  show_shapes=True, show_layer_names=True)


    def load_base_model(self, path):
        """
        Load the base model for sync-aware training
        """
        model_path = os.path.join(path, 'trained_model_base.h5')
        if os.path.isfile(model_path):
            self._model = load_model(model_path)
            self._is_sync_training = True
            # Reduce learning rate for fine-tuning
            self._learning_rate *= 0.1
            self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self._learning_rate)
            self._model.compile(loss=losses.mean_squared_error, optimizer=self.optimizer)
            return True
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


class TestModel:
    def __init__(self, input_dim, model_path, phase='base'):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path, phase)


    def _load_my_model(self, model_folder_path, phase):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, f'trained_model_{phase}.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit(f"Model not found for phase: {phase}")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim