import cv2 
import os
import glob
from shutil import copyfile

image_list = [] 
t = 0
t1 = 1
for sub in os.listdir('./TrainVal'):
    subpath = 'TrainVal/' + sub
    for filename in glob.glob(subpath + '/*.jpg'):
        try:
            print(t)
            t += 1
            im=cv2.imread(filename) 
            print(im.shape)
        except:
            t1 += 1
            print(filename)
            os.remove(filename)
        #t = filename.split('/')[-1]
        #copyfile(filename, 'train/' + t)
print(t)
print(t1)