#!/usr/bin/python
# coding = utf-8

import os
import pickle
import keras
import functools

from keras import applications
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Structure of data directories expected
# base_dir/
# ├── train/
# |   ├── class1
# |   |   ├── class1_train_img1.jpg
# |   |   ├── class1_train_img2.jpg
# |   |   └── ...
# |   ├── class2
# |   |   ├── class2_train_img1.jpg
# |   |   ├── class2_train_img2.jpg
# |   |   └── ...
# ├── validation/
# |   ├── class1
# |   |   ├── class1_val_img1.jpg
# |   |   ├── class1_val_img2.jpg
# |   |   └── ...
# |   └── ...
# └── weights/
#     ├── weights_file_1.h5
#     ├── weights_file_2.h5
#     └── ...
#
# Outputs the following files (with user-specified analysis ID prefixes):
#   model.h5            weights file for best-performing model
#   history.pkl         pickle file containing history of training/validation accuracy and loss rates throughout the run
#   predictions.pkl     pickle file containing predictions by model for all validation images
#   confusion.pkl       pickle file containing raw data to generate confusion matrix
#   labels.pkl          pickle file containing all class labels
#
# Note: Weights files can be downloaded at: https://github.com/fchollet/deep-learning-models/releases/

# Get information on available GPUs in system
k.tensorflow_backend._get_available_gpus()

# Set up top-3 accuracy metric
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'


################################################ USER INPUT BEGINS ###############################################

# Path to base directory containing training/validation data, etc.
base_dir = ''

# Path to directory to which output will be saved (will be created it it doesn't exist)
output_dir = ''

# Analysis number (or any other run identifier)
analysis_id = ''

################################################# RUN PARAMETERS #################################################
#   cnn                     convolutional neural network to use (options: 'vgg16', 'inceptionv3', 'densenet121') #
#   augment                 bool specifying whether data augmentation should be used                             #
#   reg                     bool specifying whether L1/L2 regularization should be used                          #
#   img_width, img_height   width and height in pixels that input images will be resized to                      #
#   batch_size              number of images in each feedforward batch (limited by memory availability)          #
#   epochs                  number of epochs to run training                                                     #
#   lrate                   learning rate                                                                        #
#   adjust_lrate            bool that specifies whether learning rate should be automatically adjusted           #
#   drop                    dropout parameter (= proportion of features to drop)                                 #
#   lmbda                   lambda parameter of L1/L2 regularization                                             #
#   num_feat                number of augmentation 'treatments' to use (options: 2 or 5)                         #
#   num_classes             total number of classes                                                              #
#   num_train_samples       total number of training samples                                                     #
#   num_validation_samples  total number of validation samples                                                   #
##################################################################################################################
cnn = 'vgg16'
augment = False
reg = False
img_width, img_height = 160,160
batch_size = 200
epochs = 50
lrate = 0.0001
adjust_lrate = True
drop = 0.5
lmbda = 0.01
num_feat = 2
num_classes = 36
num_train_samples = 27737
num_validation_samples = 6903
################################################ USER INPUT ENDS #################################################


# Set paths for training and validation data
train_data_dir = os.path.join(base_dir,'train')
validation_data_dir = os.path.join(base_dir,'validation')

# Determine number of steps required to send all validation/training images
# through one forward propogation. Note that if modulo(number of samples / batch size) != 0,
# an extra step is required to pass the remainder images through the network
if num_validation_samples % batch_size == 0:
    validation_steps = num_validation_samples / batch_size
else:
    validation_steps = num_validation_samples / batch_size + 1

if num_train_samples % batch_size == 0:
    train_steps = num_train_samples / batch_size
else:
    train_steps = num_train_samples / batch_size + 1

# Set weights and initialize models depending on chosen CNN
# include_top is False because we want to add change the size of the final fully-connected
# layer to match the number of classes in our specific problem
if cnn == 'vgg16':
    weights = os.path.join(base_dir,'weights','vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model = applications.VGG16(weights=weights, include_top=False, input_shape = (img_width, img_height, 3))
    layer_freeze = 7

if cnn == 'inceptionv3':
    weights = os.path.join(base_dir,'weights','inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model = applications.InceptionV3(weights=weights, include_top=False, input_shape = (img_width, img_height, 3))
    layer_freeze = 249

if cnn == 'densenet121':
    weights = os.path.join(base_dir,'weights','densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model = applications.DenseNet121(weights=weights, include_top=False, input_shape = (img_width, img_height, 3))
    layer_freeze = 313

# Now add additional layers for fine-tuning, regularization, dropout, Softmax, etc.
# Freeze early layers (up to layer specified in layer_freeze) while allowing deeper layers
# to remain trainable for fine-tuning
for layer in model.layers[:layer_freeze]:
    layer.trainable = False

x = model.output
x = Flatten()(x)

# L1/L2 regularization
if reg:
    x = Dense(1024,
              activation="relu",
              kernel_regularizer=regularizers.l2(lmbda),
              activity_regularizer=regularizers.l1(lmbda))(x)

x = Dense(1024, activation="relu")(x)

# Dropout
if drop:
    x = Dropout(drop)(x)
    x = Dense(1024, activation="relu")(x)

# Fully-connected layer for classification
predictions = Dense(num_classes, activation="softmax")(x)

# Finally, we connect the input model layers with the output fully-connected layer and compile
model_final = Model(inputs = model.input, outputs = predictions)
model_final.summary()
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=lrate), metrics=['accuracy',top3_acc])

# Data augmentation (if set) and data generators to read and process training/validation images
if augment:
    if num_feat == 5:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                            rotation_range = 20,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            fill_mode = 'nearest')
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                            rotation_range = 20,
                                            zoom_range = 0.2,
                                            fill_mode = 'nearest')
else:
    train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        class_mode = "categorical")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Set up checkpointing (saves best-performing model weights after each epoch)
checkpoint = ModelCheckpoint(os.path.join(output_dir,'analysis_{:s}_checkpoint.h5'.format(analysis_id)),
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

# Set up early stopping monitor (will stop run if validation accuracy doesn't improve for 10 epochs)
early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=10,
                      verbose=1,
                      mode='auto')

# Set up automatic learning rate adjustment if requested
if adjust_lrate:
    reduceLR = ReduceLROnPlateau(monitor = 'val_acc',factor=0.5,
                                   patience = 3, verbose = 1, mode = 'auto',
                                   min_delta = 0.005, min_lr = 0.00001)
    callbacks = [checkpoint, early, reduceLR]
else:
    callbacks = [checkpoint, early]

# Run model training using generator
history = model_final.fit_generator(
            train_generator,
            steps_per_epoch = train_steps,
            epochs = epochs,
            validation_data = validation_generator,
            validation_steps = validation_steps,
            callbacks = callbacks)

# Save best-performing model
model_final.save(os.path.join(output_dir,'analysis_{:s}_model.h5'.format(analysis_id)))

# Save histories
with open(os.path.join(output_dir,'analysis_{:s}_history.pkl'.format(analysis_id)), 'wb') as f:
    pickle.dump(history.history,f)

# Save confusion matrix, classification report, and label map
Y_pred = model.predict_generator(validation_generator, validation_steps)
y_pred = np.argmax(Y_pred, axis=1)
confusion = confusion_matrix(validation_generator.classes, y_pred)
label_map = (validation_generator.class_indices)
labels = sorted(label_map.keys())
report = classification_report(validation_generator.classes, y_pred, target_names=labels)

with open(os.path.join(output_dir,'analysis_{:s}_predictions.pkl'.format(analysis_id)),'wb') as handle:
    pickle.dump(report,handle)
with open(os.path.join(output_dir,'analysis_{:s}_confusion.pkl'.format(analysis_id)),'wb') as handle:
    pickle.dump(confusion,handle)
with open(os.path.join(output_dir,'analysis_{:s}_labels.pkl'.format(analysis_id)),'wb') as handle:
    pickle.dump(label_map,handle)
