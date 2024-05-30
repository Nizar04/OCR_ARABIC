# -*- coding: utf-8 -*-


from ocr_1 import *
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import RandomState
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import string
from shutil import copyfile, rmtree
import re
import cv2
from PIL import Image, ImageDraw
import glob

save_path = "C:/Users/nizar/OCR2"
path = "C:/Users/nizar/OCR2/enit_ifn database/data/set_a/tif"
model_name = "MODEL_DIRECTORY"
args=["Fichier_Vide"]



batch_size = 64
imgh = 100
imgw = 300

try:
    rmtree(save_path + "/" + model_name)
except:
    pass

os.mkdir(save_path + "/" + model_name)
with open(save_path + "/" + model_name + "/arguments.txt", "w") as f:
    f.write(str(args))

prng = RandomState(32)

train = [dp + "/" + f for dp, dn, filenames in os.walk(path)
         for f in filenames if re.search('tif', f)]

prng.shuffle(train)
lexicon = get_lexicon_2(train)
classes = {j: i for i, j in enumerate(lexicon)}
inve_classes = {v: k for k, v in classes.items()}

length = len(train)
train, val = train[:int(length * 0.9)], train[int(length * 0.9):]
lenghts = get_lengths(train)
max_len = max(lenghts.values())

objet = Readf(classes=classes)
y = objet.get_labels(train)

X, Y, input_len, label_len = objet.get_blank_matrices()

img_w, img_h = 300, 100
output_size = len(classes) + 1

crnn = CRNN(img_w, img_h, output_size, max_len)
model = crnn.model

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['acc'])


train_generator = objet.run_generator(train)
val_generator = objet.run_generator(val)

train_steps = len(train) // batch_size
val_steps = len(val) // batch_size

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=train_steps,
                              validation_data=val_generator,
                              validation_steps=val_steps,
                              epochs=130)



plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the model
model.save(save_path + "/" + model_name + "/trained_ocr.h5")











