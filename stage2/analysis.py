import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# load MNIST dataset
(x_train, y_train), *_ = tf.keras.datasets.mnist.load_data()

# reshape to total_image x total_pixel
total_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], total_pixels)

# split into train and test sets
NUM_OF_ROWS = 6000
TEST_SIZE = 0.3
RAND_SEED = 40
x_train, x_test, y_train, y_test = train_test_split(x_train[:NUM_OF_ROWS], y_train[:NUM_OF_ROWS],
                                                    test_size=TEST_SIZE, random_state=RAND_SEED)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("Proportion of samples per class in train set:")
print(pd.Series(y_train).value_counts(normalize=True))
