import cv2 
import os
import glob
from shutil import copyfile
import random

for sub in os.listdir('./TrainVal'):
    try:
        s.makedirs('train/' + sub)
        s.makedirs('val/' + sub)
    except:
        continue

for sub in os.listdir('./TrainVal'):
    threshold = len(os.listdir('TrainVal/'+ sub))
    val_set = random.sample(range(0, threshold), int(threshold/10))
    print(threshold, sub)
    print(sub)
    t = 0
    subpath = 'TrainVal/' + sub
    for filename in glob.glob(subpath + '/*.jpg'):
        im=cv2.imread(filename)
        if t in val_set:
            cv2.imwrite('val/' + sub + '/' + filename.split('/')[-1], im)
        else:
            cv2.imwrite('train/' + sub + '/' + filename.split('/')[-1], im)
        t += 1

