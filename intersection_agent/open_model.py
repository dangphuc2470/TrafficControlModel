import tensorflow as tf

# Load the model using TensorFlow 2.x in the SavedModel format
loaded_model = tf.keras.models.load_model(r"F:\P\HK8\Deep-QLearning-Agent-for-Traffic-Signal-Control\TLCS\models\model_5\trained_model.h5")