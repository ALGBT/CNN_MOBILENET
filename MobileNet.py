# ---- Imports ---- #

import random
import numpy as np
import pickle
from keras.applications import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.models import Model
import keras
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import MaxPool2D
import numpy as np
import os
import tensorflow
import cv2

# ---- End of Imports ---- #

# ---- Basic Parameters ---- #

EPOCHS = 15 # Number of Epochs
INIT_LR = 1e-5  # Initial learning rate
BS = 16 # Batch size, Currently 8
default_image_size = tuple((224, 224))  # Default image size, all the images would be reshaped to this size
image_size = 0  # Current image size
directory_root = 'plantvillage dataset'  # Dataset Directory
width = 224  # Width
height = 224  # Height
depth = 3  # RGB-> Depth = 3
labels_to_write = []

# ---- End of Basic Parameters ---- #

# ---- Callback class, allows us to modify our NN and inputs after every epoch ---- #

class UpdateFunctionParams(keras.callbacks.Callback):

    # ---- Consturctor function for the class ---- #

    def __init__(self):
        super(UpdateFunctionParams, self).__init__() # Use the UpdateFunctionParams constructor
        self.num = 0 # Num attribute to count how many epochs we already had
        self.epochs = [0] # keep a list of epochs
        self.losses = [0] # Keep a list of loss values
        self.noise_history = 0 # Save the noise we already added

        # ---- End of constructor function ---- #

        # ---- The function is called after each epoch of the NN and saves the wanted information from the epoch ---- #

    def on_epoch_end(self, epoch, logs=None):
        self.num=self.num+1 # Save the next epoch num
        self.epochs.append(self.num) # Save the epoch we had num
        self.losses.append((logs.get('val_loss'))) # Save the loss values of the epoch

        # ---- End of the on_epoch_end function --- #

# ---- Function add_noise, the function receives an image and add noise to this image ---- #

def add_noise(img):
    epoch_num = call.num # Get the epoch num
    epoch_add = EPOCHS - call.num - 5 # Needed info for noise calculation
    if(epoch_num !=0): # We don't add noise at the first epoch
        point1 = np.array([call.epochs[epoch_num - 1], call.losses[epoch_num - 1]]) # get the info from the previous epoch
        point2 = np.array([call.epochs[epoch_num], call.losses[epoch_num]]) # Get the info from the current epoch
        adding = (epoch_add) / 100 # Calculate the wanted added noise(we decided to take that based on epoch added)
        if(np.linalg.norm(point2 - point1) < 0 and call.noise_history < 0.8): # check if we reached our noise limit(0.8)
            call.noise_history = call.noise_history + adding # if not add the noise to our noise history
    noise = np.random.normal(loc=0, scale=1, size=img.shape) # Random the wanted noise from gaussian distribution
    img2 = img * 2 # Layer the image
    # add the noise to our image using the layerd image and our random noise * our wanted added noise
    n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * call.noise_history)),2*(1 - img2 + 1) * (1 + noise * call.noise_history) * -1 + 2) / 2, 0, 1)
    return n2 # Return the noised image

# ---- End of function ---- #

# ---- Function convert_image_to_array turns the images we have into an np array ---- #

def convert_image_to_array(image_dir):
    try:  # Try to read the image and reshape it
        image = cv2.imread(image_dir)  # Read the image
        if image is not None:  # If the image is not blank
            image = cv2.resize(image, default_image_size)  # Resize it to the shape wanted
            return img_to_array(image)  # Return the image as
        else:  # If the image is blank return an empty np array
            return np.array([])
    except Exception as e:  # If an exception was raised notify us with it
        print(f"Error : {e}")
        return None

# ---- End of the function, images converted ---- #

# ---- Loading the images we would like to use ---- #

image_list, label_list = [], []  # List to hold classes and images
try:
    print("[INFO] Loading images ...")
    root_dir = os.listdir(directory_root) # Save the image directory

    # ---- Iterate over subfolders in our root folder ---- #

    for plant_folder in root_dir:
        plant_disease_folder_list = os.listdir(f"{directory_root}/{plant_folder}") # Save the paths of all the subfolders

        # ---- Iterate over each subfolder subfolder ----- #

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = os.listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/") # Save the paths to the images
            labels_to_write.append(plant_disease_folder) # Save the label

            # ---- Iterate over the images ---- #

            for image in plant_disease_image_list[:1400]: # Load only 1400 due to resources
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}" # save the current file
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True: # Check if image
                    image_list.append(convert_image_to_array(image_directory)) # if so save it
                    label_list.append(plant_disease_folder) # Save the label of the image
    print("[INFO] Image loading completed")

except Exception as e:
    print(f"Error : {e}")

# ---- End of image loading ---- #

image_size = len(image_list)
label_binarizer = LabelBinarizer() # Create a Labelbinraizer object
image_labels = label_binarizer.fit_transform(label_list) # Save the labels
n_classes = len(label_binarizer.classes_)
with open('labels_new.txt', 'w') as f: # Save the labels in a txt file
    for line in labels_to_write:
        f.write(f"{line}\n")
np_image_list = np.array(image_list, dtype=np.float16) / 225.0 # Flatten the images
print("Hello,List len", len(np_image_list))
print("[INFO] Spliting data to train, test")
X_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42) # Split to train and test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # Split the train to train and validation
print("Hello, X_train len", len(X_train))

# ---- Data augmentation object ---- #

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest",
    # brightness_range=[0.7, 1],
    # channel_shift_range=30
    preprocessing_function=add_noise, # Use the add_noise function
)

# ---- End of data augmentation object ---- #

# ---- Image manipulation, if the image is not in the wanted format and with channel first than chanDim = 1 ---- #

inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1

# ---- End of image manipulation ---- #


# ---- Creating the model ---- #
model_finetuned = Sequential()  # The model is sequential

model_finetuned.add(MobileNetV2(weights='imagenet'))  # Add MobileNet Architecture with imagenet weights
model_finetuned.add(BatchNormalization())  # Add BatchNormalization to the network
model_finetuned.add(Dense(128, activation="relu"))  # First classifier layer
model_finetuned.add(Dense(49, activation="softmax"))  # Classifier output layer
for layer in model_finetuned.layers[
    0].layers:  # Check if the layer is trainable or not, meaning should we freeze it or should train it
    if layer.__class__.__name__ == "BatchNormalization":
        layer.trainable = True
    else:
        layer.trainable = False
model_finetuned.compile(optimizer='adam',
                        loss=tensorflow.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])

# ---- End of model creation ---- #
print("[INFO] training network...")
call = UpdateFunctionParams()
# ---- Train the model ---- #
# ---- Aug.flow(create new images using Data augmentation, validation_data = the validation set ---- #
history = model_finetuned.fit_generator(
    aug.flow(X_train, y_train, batch_size=BS),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // BS,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[call])

# ---- Train ended ---- #

# ---- Evaluate the model on the test set ---- #
print("[INFO] Calculating model accuracy")
scores = model_finetuned.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1] * 100}")
# ---- Evaluation ended ---- #

# ---- Save the model in pickle format ---- #
print("[INFO] Saving model...")
pickle.dump(model_finetuned, open('DiseaseDetectionCNN_mobile_111', 'wb'))

# ---- Save the model using Keras.save ---- #
model_finetuned.save("DiseaseDetectionCNN_V333.h5")

# ---- Create a JSON file with the model data ---- #
model_json = model_finetuned.to_json()
with open("model_Mobile.json22222", "w") as json_file:
    json_file.write(model_json)

# ---- serialize weights to HDF5 ---- #
model_finetuned.save_weights("model_Mobile3434334.h5")
print("Saved model to disk")
