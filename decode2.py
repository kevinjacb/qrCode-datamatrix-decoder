from base64 import decode
from pyzbar.pyzbar import decode as qrDecode
from pylibdmtx.pylibdmtx import decode as dmDecode
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model

'''
    In order to use this function first execute:
    pip install pyzbar, pylibdmtx
'''

path = 'test_qr/'


def decodeImage(image) -> str:
    qrInfo = qrDecode(image)
    dmInfo = dmDecode(image)

    if(not len(qrInfo) and not len(dmInfo)):
        return None

    if(not len(qrInfo)):
        box = dmInfo[0][1]
        # cv.imshow("data matrix", drawBoundingBox(image, box))
        return dmInfo[0][0].decode('utf-8')
    else:
        box = qrInfo[0][1]
        # cv.imshow("qr_code", drawBoundingBox(image, box))
        return qrInfo[0][0].decode('utf-8')


kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

kernel_sharpen2 = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel('EDSR_x4.pb')
sr.setModel('edsr', 4)

model = tf.keras.models.load_model('qr_model_weights.h5')

for i, filename in enumerate(os.listdir(path)):
    print(filename)
    if not filename.endswith('.jpg'):
        continue
    img = cv.imread(path+filename)
    res = decodeImage(img)
    if(res == None):
        img_pred = np.array(img)/255.0
        box = np.array(model.predict(
            img_pred.reshape(-1, 480, 640, 3))[0], dtype=np.int16)
        print(box)
        actual_roi = cv.rectangle(
            img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 3)
        roi = img[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]
        img_sr = sr.upsample(roi)
        img_sharpened = cv.filter2D(img_sr, -1, kernel_sharpen2)
        res = decodeImage(img_sharpened)
        cv.imwrite(f'output/{i}.jpg', img_sharpened)
    print(f'{i} file: {filename} data: {res}')
