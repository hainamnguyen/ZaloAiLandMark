from __future__ import print_function
import os.path
import csv
import glob
import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import ResNet50, Xception, InceptionResNetV2
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing import image

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
IMAGE_SIZE    = (480, 480)
NUM_CLASSES   = 103

base_model = ResNet50(include_top=False,
                        weights=None,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('weights_trained/resnet50.h5', by_name=True)

class_order = ['0', '1', '10', '100', '101', '102', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('/home/phoenix/Hana/challenge/Public/',
        target_size=(480, 480),
        shuffle = False,
        class_mode='categorical',
        batch_size=32)
filenames = test_generator.filenames
filenames = [i.split("/")[-1].split('.')[0] for i in filenames]
predicts_nohot = model.predict_generator(test_generator)

np.savez('result/resnet50.npz', predicts_nohot=predicts_nohot)

results = []
for i in predicts_nohot:
    result_raw = i.argsort()[::-1][:3]
    class_name = [str(class_order[t]) for t in result_raw]
    results.append(" ".join(class_name))

w = csv.writer(open("result_resnet50.csv", "w"))
w.writerow(['id', 'predicted'])
for i in range(len(results)):
    #print(key, value)
    w.writerow([filenames[i], results[i]])
