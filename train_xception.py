from __future__ import print_function

import os.path

import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import ResNet50, Xception, InceptionResNetV2
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

DATASET_PATH_train  = '/home/phoenix/Hana/challenge/TrainVal'
DATASET_PATH_val  = '/home/phoenix/Hana/challenge/val'

# DATASET_PATH  = 'C:\\Users\\kccs\\Downloads\\data_201804_datasets_color\\data_201804_datasets_color'
IMAGE_SIZE    = (480, 480)
NUM_CLASSES   = 103
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 0  # freeze the first this many layers for training
NUM_EPOCHS    = 10
WEIGHTS_FINAL = 'model-exception_v2-final.h5'

tensorboard_directory   = 'logs_05'
weight_directory        = 'weights_exception'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH_train,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH_val,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

#print(DATASET_PATH)
# show class indices
print('****************')
classes_name = [0 for i in range(103)]
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
    classes_name[idx] = cls
print(classes_name)
print('****************')

# build our classifier model based on pre-trained InceptionResNetV2:
# 1. we don't include the top (fully connected) layers of InceptionResNetV2
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
base_model = Xception(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
for layer in base_model.layers:
    layer.trainable = True

# add a global spatial average pooling layer

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('weights_exception/weights.10.h5', by_name=True)
# for layer in model.layers[:126]:
#     layer.trainable = False
#for layer in model.layers[126:]:
    #layer.trainable = True

# net_final.compile(optimizer=Adam(lr=1e-5),
#                   loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-9, amsgrad=True), loss='categorical_crossentropy',
                  metrics=[top_3_accuracy, 'accuracy'])

print(model.summary())

tensorboard_callback = TensorBoard(log_dir=tensorboard_directory, 
                                                       histogram_freq=0,
                                                       write_graph=True,
                                                       write_images=False)
save_model_callback = ModelCheckpoint(os.path.join(weight_directory, 'weights.{epoch:02d}.h5'),
                                                        verbose=3,
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)

# train the model
model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches, 
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        callbacks=[save_model_callback, tensorboard_callback],
                        epochs = NUM_EPOCHS)

# save trained weights
model.save(WEIGHTS_FINAL)


'''
import h5py
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

#dataset = pickle.load( open( "dataset1.pkl", "rb" ) )
dataset = h5py.File('dataset.hdf5', 'r')

x_train = dataset["train_img"]
for i in range(len(x_train)):
    i +=
y_train = dataset["train_labels"]


# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet',input_shape=(480, 480, 3), include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['top_3_accuracy'])

model.fit_generator(...)
'''

'''
https://www.kaggle.com/nothxplz/keras-inception-resnet-inception-resnet50/code
'''

