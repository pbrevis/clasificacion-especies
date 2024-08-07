# El siguiente código fue ejecutado en un 'notebook' de Google Colab

# Mounting Google Drive

from google.colab import drive
drive.mount('/content/gdrive')

# Install packages

from tensorflow.keras.layers import Conv2D,  Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import random
import os

# Load images

img_height, img_width = (224,224)
batch_size = 128


train_data_dir = "/content/gdrive/MyDrive/carabid-detector/training"
test_data_dir = "/content/gdrive/MyDrive/carabid-detector/testing"

save_dir = "/content/gdrive/MyDrive/carabid-detector/my_save_folder"

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    seed = 123,
    shuffle = True,
    class_mode = 'categorical')


test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    seed = 123,
    shuffle = False,
    class_mode = 'categorical')

os.chdir(save_dir)

# Set seeds

seed_value= 321

os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

# Define & fit model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_weights_CNN.weights.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_generator,
          epochs = 100,
          validation_data = test_generator,
          callbacks = [early_stopping, checkpoint])

model.save('my_model.keras')

# Make and save predictions

best_model = tf.keras.models.load_model('/content/gdrive/MyDrive/carabid-detector/my_save_folder/my_model.keras')

np.random.seed(seed_value)
tf.random.set_seed(seed_value)

import pandas as pd

preds = model.predict(test_generator)
preddf = pd.DataFrame(preds)
preddf.to_csv("CNN-Predictions.csv", index = False)
