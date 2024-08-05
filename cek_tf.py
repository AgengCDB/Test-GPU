import tensorflow as tf

print("Checking tensorflow version...")
print(tf.__version__)

# Check if TensorFlow is using the GPU
print("Checking tensorflow is using the GPU...")
print(tf.config.list_physical_devices('GPU'))
