import numpy as np
import csv
from keras.preprocessing.image import ImageDataGenerator

class_order = ['0', '1', '10', '100', '101', '102', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

result_exception = np.load('result/xception.npz')
result_exception = result_exception['predicts_nohot']

result_inceptionresnet = np.load('result/inception_resnet.npz')
result_inceptionresnet = result_inceptionresnet['predicts_nohot']

result_resnet101 = np.load('result/resnet101.npz')
result_resnet101 = result_resnet101['predicts_nohot']

result_resnet50 = np.load('result/resnet50.npz')
result_resnet50 = result_resnet50['predicts_nohot']

result_raw = result_exception + result_inceptionresnet + result_resnet101 + result_resnet50
#result_raw = result_exception + result_inceptionresnet + result_resnet50

results = []

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('/home/phoenix/Hana/challenge/Public/',
        target_size=(480, 480),
        shuffle = False,
        class_mode='categorical',
        batch_size=32)
#print('test_generator.filenames', test_generator.filenames)
filenames = test_generator.filenames
filenames = [i.split("/")[-1].split('.')[0] for i in filenames]


for i in result_raw:
    result_ = i.argsort()[::-1][:3]
    class_name = [str(class_order[t]) for t in result_]
    results.append(" ".join(class_name))

w = csv.writer(open("result/finalresult.csv", "w"))
w.writerow(['id', 'predicted'])
for i in range(len(results)):
    #print(key, value)
    w.writerow([filenames[i], results[i]])