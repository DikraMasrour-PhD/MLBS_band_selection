import numpy as np
import tensorflow as tf
import time

# Defining constants
DATASET_NAME = "IP"

# Model and dataset addresses
MODEL_ADDRESS = "hbs_mlbs_"+DATASET_NAME
DATASET_ADDRESS = ["hbs_mlbs_"+DATASET_NAME+"\\image_train_"+DATASET_NAME+".npy",
                   "hbs_mlbs_"+DATASET_NAME+"\\label_train_"+DATASET_NAME+".npy",
                   "hbs_mlbs_"+DATASET_NAME+"\\image_test_"+DATASET_NAME+".npy",
                   "hbs_mlbs_"+DATASET_NAME+"\\label_test_"+DATASET_NAME+".npy"]

# Loading the training dataset.
model = tf.keras.models.load_model(MODEL_ADDRESS)
image_train = np.load(DATASET_ADDRESS[0])
label_train = np.load(DATASET_ADDRESS[1])
label_train_int = np.argmax(label_train, axis=1)

start_time = time.time()
model.predict(image_train)
# model.predict([image_train[0][np.newaxis], image_train[0][np.newaxis]])
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 