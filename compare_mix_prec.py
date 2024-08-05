import tensorflow as tf
import numpy as np
import time
from colorama import Fore, Style

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Mixed Precision Policy
    if tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data():
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y_train = y_train.astype(np.int32)
    return x_train, y_train

def measure_training_time_accuracy_loss(device_name):
    with tf.device(device_name):
        model = build_model()
        x_train, y_train = load_data()

        start_time = time.time()
        
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.epoch_start_time = None
                self.start_time = time.time()
                self.epochs = 5
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                # print(f"\n{Fore.YELLOW}Epoch {epoch+1}/{self.epochs} - Training Started{Style.RESET_ALL}")
            
            def on_epoch_end(self, epoch, logs=None):
                elapsed_time = time.time() - self.epoch_start_time
                print(f"\r{Fore.YELLOW}Epoch {epoch+1}/{self.epochs} - Elapsed time: {elapsed_time:.2f} seconds{Style.RESET_ALL}", end='')
        
        progress_callback = ProgressCallback()

        # Increase batch size for GPU
        batch_size = 64 if device_name == '/GPU:0' else 32

        history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size, verbose=0, callbacks=[progress_callback])

        end_time = time.time()
        training_time = end_time - start_time

        accuracy = history.history['accuracy'][-1]
        loss = history.history['loss'][-1]
        
        return training_time, accuracy, loss

def display_comparison(cpu_time, cpu_accuracy, cpu_loss, gpu_time, gpu_accuracy, gpu_loss):
    print()
    
    print(f"\n{Fore.GREEN}Training time on CPU: {cpu_time:.2f} seconds{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Accuracy on CPU: {cpu_accuracy:.8f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Loss on CPU: {cpu_loss:.8f}{Style.RESET_ALL}")

    print()

    if gpu_time is not None:
        print(f"{Fore.CYAN}Training time on GPU: {gpu_time:.2f} seconds{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Accuracy on GPU: {gpu_accuracy:.8f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Loss on GPU: {gpu_loss:.8f}{Style.RESET_ALL}")

        print()

        speedup = cpu_time / gpu_time
        time_diff = cpu_time - gpu_time

        if time_diff > 0:
            faster_device = "GPU"
        else:
            faster_device = "CPU"
            time_diff = -time_diff

        print(f"{Fore.MAGENTA}Speedup: GPU is {speedup:.2f} times faster than CPU{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Time Difference: {faster_device} is faster by {time_diff:.2f} seconds{Style.RESET_ALL}")
        
        try:
            if cpu_accuracy > gpu_accuracy:
                print(f'{Fore.MAGENTA}CPU accuracy is better by {(cpu_accuracy - gpu_accuracy):.8f}{Style.RESET_ALL}')
            else:
                print(f'{Fore.MAGENTA}GPU accuracy is better by {(gpu_accuracy - cpu_accuracy):.8f}{Style.RESET_ALL}')
        except Exception as e:
            print(e)
        
        try:
            if cpu_loss > gpu_loss:
                print(f'{Fore.MAGENTA}CPU loss is better by {(cpu_loss - gpu_loss):.8f}{Style.RESET_ALL}')
            else:
                print(f'{Fore.MAGENTA}GPU loss is better by {(gpu_loss - cpu_loss):.8f}{Style.RESET_ALL}')
        except Exception as e:
            print(e)
    else:
        print(f"{Fore.RED}No GPU available{Style.RESET_ALL}")

cpu_time, cpu_accuracy, cpu_loss = measure_training_time_accuracy_loss('/CPU:0')

if tf.config.list_physical_devices('GPU'):
    gpu_time, gpu_accuracy, gpu_loss = measure_training_time_accuracy_loss('/GPU:0')
    display_comparison(cpu_time, cpu_accuracy, cpu_loss, gpu_time, gpu_accuracy, gpu_loss)
else:
    display_comparison(cpu_time, cpu_accuracy, cpu_loss, None, None, None)