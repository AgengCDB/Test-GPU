#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:06:32 2017

ARTIFICIAL NEURAL NETWORK

@author: Ilaria - @edited by Adrian
"""

# part 1 data pre-processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ensure TensorFlow is using the GPU
import tensorflow as tf

# Enable memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found")

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# One-hot encoding for categorical variables
categorical_features = [1, 2]
column_transformer = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X = np.array(column_transformer.fit_transform(X), dtype=np.float32)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part2: make the ANN
# import keras libraries and required packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add(Dense(units=13, kernel_initializer='uniform', activation='relu', input_dim=X.shape[1]))

# adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN - apply stochastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# defining early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.2, callbacks=[early_stopping])

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
