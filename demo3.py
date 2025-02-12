import tensorflow as tf

# Check available GPUs
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Run a small test to confirm GPU usage
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:\n", c.numpy())
