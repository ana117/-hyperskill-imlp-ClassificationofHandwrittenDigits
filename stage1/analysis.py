import tensorflow as tf
import numpy as np

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape to total_image x total_pixel
total_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], total_pixels)

x_train_min = x_train.min()
x_train_max = x_train.max()

print(f"Classes: {np.unique(y_train)}")
print(f"Features' shape: {x_train.shape}")
print(f"Target's shape: {y_train.shape}")
print(f"min: {x_train_min:.1f}, max: {x_train_max:.1f}")
