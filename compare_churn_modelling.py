import tensorflow as tf
import numpy as np
import pandas as pd
import time
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("training_log.txt")

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data():
    data = pd.read_csv('Churn_Modelling.csv')
    X = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]

    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_features = ['Geography', 'Gender']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X = preprocessor.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def measure_training_time_accuracy_loss(device_name):
    with tf.device(device_name):
        x_train, x_test, y_train, y_test = load_data()
        model = build_model(input_shape=x_train.shape[1:])

        start_time = time.time()
        
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.epoch_start_time = None
                self.start_time = time.time()
                self.epochs = 5
                self.total_steps = len(x_train) // 32
                self.current_epoch = 0
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                self.current_epoch = epoch
            
            def on_epoch_end(self, epoch, logs=None):
                elapsed_time = time.time() - self.epoch_start_time
                print(f"\r{Fore.YELLOW}Epoch {self.current_epoch+1}/{self.epochs} - Elapsed time: {elapsed_time:.2f} seconds{Style.RESET_ALL}", end='')
            
            def on_batch_end(self, batch, logs=None):
                elapsed_time = time.time() - self.start_time
                step_progress = f"{Fore.CYAN}Epoch {self.current_epoch+1}/{self.epochs} - Step {batch+1}/{self.total_steps} - Elapsed time: {elapsed_time:.2f} seconds{Style.RESET_ALL}"
                print(f"\r{step_progress}", end='')

        progress_callback = ProgressCallback()

        history = model.fit(x_train, y_train, epochs=5, verbose=0, callbacks=[progress_callback])

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
